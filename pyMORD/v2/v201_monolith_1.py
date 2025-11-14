import numpy as np
from scipy.stats import chi2, f as finv
from scipy.optimize import minimize
from scipy.io import loadmat, savemat
from scipy.signal import butter, filtfilt, iirnotch
import matplotlib.pyplot as plt
import os
import warnings
from typing import Dict, List, Tuple
from tqdm import tqdm
import glob
import h5py 

# --- 0. Unified Configuration Block ---
class Config:
    # --- I. Execution Control ---
    # Set to False to skip the hours-long optimization step and use pre-defined protocols for visualization
    RUN_OPTIMIZATION = False
    # Define a few interesting protocols from your previous run to visualize
    # Format: [Mmin, Mstep, Mmax, NDC, alpha]
    PROTOCOLS_TO_VISUALIZE = [
        [20, 40, 240, 5, 0.05],   # Example from your log
        [10, 23, 240, 8, 0.05],   # Another example
        [10, 50, 240, 3, 0.05]    # A high-speed protocol
    ]

    # --- II. Data and Paths ---
    DATA_PATH = "C:\\Users\\alexa\\experimental_data\\todos\\ENTRADAS_PATRICIA"
    OUTPUT_DIR = "./protocol_visualizations/"
    
    # --- III. Core Parameters ---
    # Reduced run counts for faster debugging/development if RUN_OPTIMIZATION is True
    N_RUNS_OPTIMIZATION = 100
    N_RUNS_VC_MSC = 1000
    ALPHA_DEFAULT = 0.05
    FP_DESEJADO_DEFAULT = 0.05
    TJ_DEFAULT = 32 # Corresponds to Fs/Tj = 1024/32 = 32 Hz resolution
    
    # --- IV. Channel and Derivation Settings ---
    TARGET_ELECTRODES = ['F3', 'F4', 'Fz', 'T5', 'T6', 'Pz', 'Cz', 'Oz']
    AVAILABLE_ELECTRODES: List[str] = ['Fz', 'F3', 'F4', 'F7', 'Fcz', 'Cz', 'C3', 'C4', 'Pz', 'P3', 'P4', 'T3', 'T4', 'T5', 'T6', 'Oz']
    
    # --- V. Signal Processing and Artifacts ---
    REMOC_THRESHOLD = 0.1 / 200 # 5e-4 V
    MAX_WINDOWS_PER_INTENSITY: Dict[str, int] = {'50dB': 240}
    SIGNAL_FREQUENCIES: List[int] = [81, 85, 89, 93, 83, 87, 91, 95]
    NOISE_FREQUENCIES: List[int] = [131, 137, 149, 151, 157, 163, 167, 173]

# --- Disabling Numba for Development ---
def njit(*args, **kwargs):
    def decorator(func):
        return func
    return decorator

# ==============================================================================
# SECTION 1: DERIVATION AND COHERENCE CALCULATION FUNCTIONS
# (Primarily from the second script)
# ==============================================================================

# @njit(float64[:,:](float64[:,:]), cache=True) # Numba disabled
def dipolos(x: np.ndarray) -> np.ndarray:
    n_samples, n_channels = x.shape
    num_combinations = n_channels * (n_channels - 1) // 2
    n_dipoles = n_channels + num_combinations
    y = np.zeros((n_samples, n_dipoles), dtype=np.float64)
    y[:, :n_channels] = x
    current_dipole_idx = n_channels
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            y[:, current_dipole_idx] = x[:, i] - x[:, j]
            current_dipole_idx += 1
    return y

def generate_dipole_labels(electrode_names: List[str]) -> List[str]:
    N = len(electrode_names)
    labels = list(electrode_names)
    for i in range(N):
        for j in range(i + 1, N):
            labels.append(f"{electrode_names[i]}-{electrode_names[j]}")
    return labels

def msweep(matrix, r):
    A = np.copy(matrix).astype(np.complex128)
    for k in range(r):
        d = A[k, k]
        if np.abs(d) < 1e-12: d = 1e-12
        col_k, row_k = A[:, k].copy(), A[k, :].copy()
        A -= np.outer(col_k, row_k) / d; A[k, :] = row_k / d; A[:, k] = -col_k / d; A[k, k] = 1.0 / d
    return A

# @njit(float64[:](complex128[:,:], int64), cache=True) # Numba disabled
def msc_fft(Y: np.ndarray, M: int) -> np.ndarray:
    sum_Y_axis1 = np.sum(Y, axis=1)
    abs_sum_Y_sq = np.abs(sum_Y_axis1)**2
    sum_abs_Y_sq_axis1 = np.sum(np.abs(Y)**2, axis=1)
    denominator = M * sum_abs_Y_sq_axis1
    ORD = np.zeros(abs_sum_Y_sq.shape, dtype=np.float64)
    for i in range(ORD.shape[0]):
        if denominator[i] != 0.0:
            ORD[i] = abs_sum_Y_sq[i] / denominator[i]
    return ORD

def _calculate_univariate_msc(data, tj, fs, alpha):
    M = data.shape[0] // tj
    if M <= 1: 
        nf = tj // 2 + 1; return np.zeros(nf), np.arange(nf) * fs / tj, 1.0
    epochs = data.flatten()[:M*tj].reshape(M, tj)
    nf = tj // 2 + 1
    Y = np.fft.fft(epochs, axis=1)[:, :nf]
    sum_Y = np.sum(Y, axis=0)
    sum_Y_sq_mag = np.sum(np.abs(Y)**2, axis=0)
    msc_values = np.zeros(nf)
    valid_idx = sum_Y_sq_mag > 1e-12
    msc_values[valid_idx] = np.abs(sum_Y[valid_idx])**2 / (M * sum_Y_sq_mag[valid_idx])
    crit = 1 - alpha**(1/(M-1))
    return msc_values, np.arange(nf) * fs / tj, crit

def _calculate_univariate_csm(data, tj, fs, alpha):
    M = data.shape[0] // tj
    if M <= 1:
        nf = tj // 2 + 1; return np.zeros(nf), np.arange(nf) * fs / tj, 1.0
    epochs = data.flatten()[:M * tj].reshape(M, tj)
    nf = tj // 2 + 1
    Y = np.fft.fft(epochs, axis=1)[:, :nf]
    teta = np.angle(Y)
    sum_cos = np.sum(np.cos(teta), axis=0)
    sum_sin = np.sum(np.sin(teta), axis=0)
    csm_values = (sum_cos**2 + sum_sin**2) / (M**2)
    crit = chi2.ppf(1 - alpha, 2) / (2 * M)
    return csm_values, np.arange(nf) * fs / tj, crit

def calculate_mmsc(data, tj, fs, alpha=0.05):
    n_samples, n_channels = data.shape if data.ndim > 1 else (data.shape[0], 1)
    if n_channels == 1: return _calculate_univariate_msc(data, tj, fs, alpha)
    M = n_samples // tj; nf = tj // 2 + 1; F = np.arange(nf) * fs / tj
    if M <= n_channels: return np.zeros(nf), F, 1.0
    epochs = data[:M*tj, :].T.reshape(n_channels, M, tj)
    Sfft = np.fft.fft(epochs, axis=2)[:, :, :nf]
    mmsc_values = np.zeros(nf)
    for kf in range(nf):
        Sfft_slice = Sfft[:, :, kf]
        spec_matrix_a = np.zeros((n_channels + 1, n_channels + 1), dtype=np.complex128)
        spec_matrix_a[:n_channels, :n_channels] = Sfft_slice @ Sfft_slice.conj().T
        V = np.sum(Sfft_slice, axis=1)
        spec_matrix_a[n_channels, :n_channels] = V.conj(); spec_matrix_a[:n_channels, n_channels] = V; spec_matrix_a[n_channels, n_channels] = M
        spec_matrix_as = msweep(spec_matrix_a, n_channels)
        mmsc_values[kf] = np.real(spec_matrix_as[n_channels, n_channels]) / M
    Fcrit = finv.ppf(1 - alpha, 2 * n_channels, 2 * (M - n_channels))
    k2Ncrit = (n_channels / (M - n_channels)) * Fcrit / (1 + (n_channels / (M - n_channels)) * Fcrit)
    return mmsc_values, F, k2Ncrit

def calculate_mcsm(data, tj, fs, alpha=0.05):
    n_samples, n_channels = data.shape if data.ndim > 1 else (data.shape[0], 1)
    if n_channels == 1: return _calculate_univariate_csm(data, tj, fs, alpha)
    M = n_samples // tj; nf = tj // 2 + 1; F = np.arange(nf) * fs / tj
    if M == 0: return np.zeros(nf), F, 1.0
    epochs = data[:M * tj, :].reshape(M, tj, n_channels); Y = np.fft.fft(epochs, axis=1)[:, :nf, :]
    teta = np.angle(Y); teta_med = np.arctan2(np.sum(np.sin(teta), axis=0), np.sum(np.cos(teta), axis=0))
    sum_cos = np.sum(np.cos(teta_med), axis=1); sum_sin = np.sum(np.sin(teta_med), axis=1)
    mcsm_values = (sum_cos**2 + sum_sin**2) / (M**2 * n_channels**2)
    csmNcrit = chi2.ppf(1 - alpha, 2 * n_channels) / (2 * M * n_channels)
    return mcsm_values, F, csmNcrit

def calculate_d_amsc(data, tj, fs, alpha=0.05):
    n_derivations = data.shape[1]; all_mscs = []
    F, crit = None, None
    for i in range(n_derivations):
        msc_values, F_temp, crit_temp = _calculate_univariate_msc(data[:, i], tj, fs, alpha)
        all_mscs.append(msc_values)
        if F is None: F, crit = F_temp, crit_temp
    return np.mean(np.array(all_mscs), axis=0), F, crit

def calculate_d_acsm(data, tj, fs, alpha=0.05):
    n_derivations = data.shape[1]; all_csms = []
    F, crit = None, None
    for i in range(n_derivations):
        csm_values, F_temp, crit_temp = _calculate_univariate_csm(data[:, i], tj, fs, alpha)
        all_csms.append(csm_values)
        if F is None: F, crit = F_temp, crit_temp
    return np.mean(np.array(all_csms), axis=0), F, crit

# ==============================================================================
# SECTION 2: PROTOCOL OPTIMIZATION FUNCTIONS
# (Primarily from the first script, used only if RUN_OPTIMIZATION is True)
# ==============================================================================

def generate_protocol_parameters(Mmax: int) -> np.ndarray:
    P = []
    for M_step in range(1, Mmax):
        for Mmin in range(2, Mmax):
            k = (Mmax - Mmin) / M_step
            if k == int(k) and k >= 0: P.append([Mmin, M_step, Mmax])
    P.append([Mmax, 1, Mmax])
    return np.array(P, dtype=np.int64)

# @njit(types.UniTuple(int64, 2)(float64[:], int64[:], float64, float64, float64[:]), cache=True) # Numba disabled
def ETS(ord_values: np.ndarray, MM_indices: np.ndarray, alfa: float, NDC: float, vc_msc_critical_values: np.ndarray) -> Tuple[int, int]:
    NDC_ceil = int(np.ceil(NDC)); det = ord_values > vc_msc_critical_values
    count, dr, time = 0, 0, MM_indices[-1]
    for ii in range(len(MM_indices)):
        if det[ii]: count += 1
        else: count = 0
        if count >= NDC_ceil:
            dr = 1; time = MM_indices[ii]; break
    return dr, time

def _monte_carlo_single_run(M_int: int, N_window: int, bin_idx: int) -> float:
    x = np.random.randn(N_window * M_int)
    aux_fft_reshaped = np.fft.fft(x.reshape(N_window, M_int), axis=0)
    relevant_fft_slice = aux_fft_reshaped[bin_idx, :].reshape(1, M_int)
    return msc_fft(relevant_fft_slice, M_int)[0]

def VC_MSC(M_values: np.ndarray, alfa: float, nRuns: int) -> Tuple[np.ndarray, np.ndarray]:
    N_window, bin_idx = 32, 7
    VC_MC = np.zeros_like(M_values, dtype=float)
    for i, M in enumerate(M_values):
        if M <= 1: VC_MC[i] = 1.0; continue
        ord_values = [_monte_carlo_single_run(int(M), N_window, bin_idx) for _ in range(nRuns)]
        VC_MC[i] = np.quantile(ord_values, 1.0 - alfa)
    VC_teorico = np.where(M_values > 1, 1 - alfa**(1.0 / (M_values - 1)), 1.0)
    return VC_MC, VC_teorico

def _generate_ord_sim(nRuns, Mmax, tj, bin_idx):
    Ntotal = Mmax * tj
    ord_sim = np.zeros((nRuns, Mmax))
    for ii in tqdm(range(nRuns), desc="Simulating Outputs for Optimization"):
        x = np.random.randn(Ntotal)
        fft_x = np.fft.fft(x.reshape(tj, Mmax), axis=0)
        for M in range(2, Mmax + 1):
            ord_sim[ii, M-1] = msc_fft(fft_x[bin_idx, :M].reshape(1, M), M)[0]
    return ord_sim

def estimarNDC(NDCinicial, alfa_teste, FP_desejado, ord_sim, Mmin, Mstep, Mmax, nRuns_vc_msc):
    MM_indices = np.arange(Mmin, Mmax + 1, Mstep)
    NNTmax = len(MM_indices)
    if NNTmax == 0: return 1.0
    _, vc_teorico = VC_MSC(MM_indices, alfa_teste, nRuns=nRuns_vc_msc)
    ord_protocol = ord_sim[:, MM_indices - 1]
    FP_values = np.zeros(NNTmax + 1)
    for NDC_val in range(1, NNTmax + 1):
        dr = np.array([ETS(ord_protocol[ii, :], MM_indices, alfa_teste, float(NDC_val), vc_teorico)[0] for ii in range(ord_sim.shape[0])])
        FP_values[NDC_val] = np.mean(dr)
        if FP_values[NDC_val] < FP_desejado: break
    ind = np.where((FP_values < FP_desejado) & (FP_values != 0))[0]
    return float(ind[0]) if ind.size > 0 else float(NNTmax)

def funcao_custo_v2(alpha, NDC, MM_indices, ord_sim, FP_desejado):
    alpha_float = np.clip(alpha[0], 1e-9, 1 - 1e-9)
    vc = np.where(MM_indices > 1, 1 - alpha_float**(1.0 / (MM_indices - 1)), 1.0)
    dr = np.array([ETS(ord_sim[ii, MM_indices - 1], MM_indices, alpha_float, NDC, vc)[0] for ii in range(ord_sim.shape[0])])
    erro = np.mean(dr) - FP_desejado
    return 0.5 * erro**2, np.array([erro])

def find_optimal_protocols(nRuns, Mmax, alfa_teste, FP_desejado, nRuns_vc_msc):
    ord_sim = _generate_ord_sim(nRuns, Mmax, Config.TJ_DEFAULT, 8) # Using bin_idx=8
    P = generate_protocol_parameters(Mmax)
    alfa_corrigido, NDC_minimo = np.full(P.shape[0], np.nan), np.full(P.shape[0], np.nan)
    num_tests = np.ceil((Mmax - P[:, 0]) / P[:, 1]) + 1
    P = P[np.argsort(num_tests), :]
    for ii in tqdm(range(P.shape[0]), desc="Optimizing Protocols"):
        Mmin, Mstep, current_Mmax = P[ii, :]
        MM_indices = np.arange(Mmin, current_Mmax + 1, Mstep)
        NDC_minimo[ii] = estimarNDC(1, alfa_teste, FP_desejado, ord_sim, Mmin, Mstep, current_Mmax, nRuns_vc_msc)
        cost_func = lambda alpha: funcao_custo_v2(alpha, NDC_minimo[ii], MM_indices, ord_sim, FP_desejado)
        res = minimize(cost_func, np.array([alfa_teste]), jac=True, method='CG', options={'maxiter': 50})
        alfa_corrigido[ii] = res.x[0]
    return np.hstack((P, NDC_minimo[:, np.newaxis], alfa_corrigido[:, np.newaxis]))

# ==============================================================================
# SECTION 3: NEW VISUALIZATION AND ANALYSIS FUNCTION
# ==============================================================================

def visualize_protocol_application(eeg_data, protocol_params, fs, title, filename, output_dir):
    Mmin, Mstep, Mmax, NDC, alpha = protocol_params
    tj = Config.TJ_DEFAULT
    num_windows_total = eeg_data.shape[0] // tj
    eeg_data = eeg_data[:num_windows_total * tj] # Ensure integer number of windows
    
    # 1. Calculate the MSC values at each step of the protocol
    MM_indices = np.arange(int(Mmin), int(Mmax) + 1, int(Mstep))
    MM_indices = MM_indices[MM_indices <= num_windows_total] # Ensure we don't exceed available data
    if MM_indices.size == 0:
        tqdm.write(f"AVISO: Protocolo {protocol_params} pulado para {title} pois não há dados suficientes ({num_windows_total} janelas).")
        return
        
    ord_values = np.zeros(len(MM_indices))
    all_deriv_data_reshaped = eeg_data.reshape(num_windows_total, tj, eeg_data.shape[1])
    xfft = np.fft.fft(all_deriv_data_reshaped, axis=1)
    
    # Calculate average MSC across all derivations for each M
    for i, M in enumerate(MM_indices):
        msc_vals_at_m = [msc_fft(xfft[:M, :, d], M) for d in range(eeg_data.shape[1])]
        avg_msc_at_m = np.mean(np.array(msc_vals_at_m), axis=0)
        # We need to pick one bin. Let's use the one from optimization.
        ord_values[i] = avg_msc_at_m[8] # bin_idx=8

    # 2. Get the theoretical critical value and perform detection
    _, vc_teorico = VC_MSC(MM_indices, alpha, nRuns=100) # Use fewer runs for speed
    detection_result, detection_time = ETS(ord_values, MM_indices, alpha, NDC, vc_teorico)

    # 3. Calculate full-spectrum coherence metrics for visualization
    d_amsc, F, crit_d_amsc = calculate_d_amsc(eeg_data, tj, fs, alpha)
    d_acsm, _, crit_d_acsm = calculate_d_acsm(eeg_data, tj, fs, alpha)

    # 4. Create the plot
    fig, axs = plt.subplots(3, 1, figsize=(18, 20), gridspec_kw={'height_ratios': [1, 1, 1]})
    fig.suptitle(title, fontsize=16)

    # Subplot 1: Protocol Decision
    axs[0].plot(MM_indices, ord_values, 'o-', color='blue', label='MSC Médio (bin 8)')
    axs[0].plot(MM_indices, vc_teorico, '--', color='red', label='Valor Crítico Teórico')
    status = f"DETECTADO em M={int(detection_time)}" if detection_result else "NÃO DETECTADO"
    axs[0].set_title(f"Decisão do Protocolo [Mmin={int(Mmin)}, Mstep={int(Mstep)}, NDC={int(NDC)}] -> {status}", fontsize=14)
    axs[0].set_xlabel("Número de Janelas (M)"); axs[0].set_ylabel("MSC"); axs[0].legend(); axs[0].grid(True, ls='--')

    # Subplot 2: D-aMSC Spectrum
    axs[1].plot(F, d_amsc, color='darkgreen', label='D-aMSC')
    axs[1].axhline(crit_d_amsc, color='k', ls='--', label=f'Limiar Crítico ({alpha*100}%)')
    axs[1].set_title("Espectro de Coerência de Magnitude (D-aMSC)", fontsize=14)
    axs[1].set_xlabel("Frequência (Hz)"); axs[1].set_ylabel("Coerência"); axs[1].legend(); axs[1].grid(True, ls='--'); axs[1].set_xlim(0, 200)

    # Subplot 3: D-aCSM Spectrum
    axs[2].plot(F, d_acsm, color='purple', label='D-aCSM')
    axs[2].axhline(crit_d_acsm, color='k', ls='--', label=f'Limiar Crítico ({alpha*100}%)')
    axs[2].set_title("Espectro de Coerência de Fase (D-aCSM)", fontsize=14)
    axs[2].set_xlabel("Frequência (Hz)"); axs[2].set_ylabel("Coerência"); axs[2].legend(); axs[2].grid(True, ls='--'); axs[2].set_xlim(0, 200)

    for ax in [axs[1], axs[2]]:
        for freq in Config.SIGNAL_FREQUENCIES: ax.axvline(x=freq, color='g', linestyle=':', alpha=0.7)
        for freq in Config.NOISE_FREQUENCIES: ax.axvline(x=freq, color='r', linestyle=':', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, filename))
    plt.close(fig)

# ==============================================================================
# SECTION 4: MAIN ANALYSIS BLOCK
# ==============================================================================

def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    param_filename = f"optimized_protocols_Mmax{Config.MAX_WINDOWS_PER_INTENSITY['50dB']}.mat"
    param_filepath = os.path.join(Config.DATA_PATH, param_filename)

    if Config.RUN_OPTIMIZATION:
        print("--- INICIANDO FASE DE OTIMIZAÇÃO DE PROTOCOLOS ---")
        parametros = find_optimal_protocols(
            nRuns=Config.N_RUNS_OPTIMIZATION, Mmax=Config.MAX_WINDOWS_PER_INTENSITY['50dB'],
            alfa_teste=Config.ALPHA_DEFAULT, FP_desejado=Config.FP_DESEJADO_DEFAULT,
            nRuns_vc_msc=Config.N_RUNS_VC_MSC)
        savemat(param_filepath, {'parametros': parametros})
        print(f"Parâmetros ótimos salvos em: {param_filepath}")
    else:
        print("--- PULANDO OTIMIZAÇÃO. USANDO PROTOCOLOS PRÉ-DEFINIDOS PARA VISUALIZAÇÃO ---")
        # If a file exists, load it. Otherwise, use the defaults from Config.
        if os.path.exists(param_filepath):
            print(f"Carregando parâmetros otimizados de: {param_filepath}")
            parametros = loadmat(param_filepath)['parametros']
            # You might want to select a few from the loaded file to visualize
            print(f"Total de {parametros.shape[0]} protocolos carregados. Visualizando os definidos em Config.")
        else:
            print("AVISO: Arquivo de parâmetros não encontrado. Usando a lista de protocolos padrão para visualização.")
        # Regardless of loading, we use the specific list for visualization
        parametros_to_visualize = Config.PROTOCOLS_TO_VISUALIZE

    # --- Main Visualization Loop ---
    all_mat_files = glob.glob(os.path.join(Config.DATA_PATH, '*.mat'))
    electrode_indices = [Config.AVAILABLE_ELECTRODES.index(e) for e in Config.TARGET_ELECTRODES]

    for file_path in tqdm(all_mat_files, desc="Processando Voluntários para Visualização"):
        try:
            filename_stem = os.path.splitext(os.path.basename(file_path))[0]
            if not filename_stem.endswith("50dB"): continue # Only process 50dB for this example

            with h5py.File(file_path, 'r') as f:
                raw_data, fs = f['x'][:].transpose(2, 1, 0), f['Fs'][0, 0] # shape -> (windows, samples, channels)
            
            # 1. Prepare Data: Get all derivations for the selected electrodes
            x_subset = raw_data[:, :, electrode_indices]
            num_windows, num_samples, num_chans = x_subset.shape
            
            # Reshape for dipolos function and apply it window by window
            all_derivations = np.zeros((num_windows, num_samples, len(generate_dipole_labels(Config.TARGET_ELECTRODES))))
            for i in range(num_windows):
                all_derivations[i, :, :] = dipolos(x_subset[i, :, :])
            
            # 2. Clean Data
            noisy_windows = np.max(np.abs(all_derivations), axis=(1, 2)) > Config.REMOC_THRESHOLD
            clean_derivations = all_derivations[~noisy_windows, :, :]
            
            # Reshape to continuous time series for analysis functions
            n_clean_windows, _, n_derivs = clean_derivations.shape
            eeg_data_for_analysis = clean_derivations.transpose(1,0,2).reshape(num_samples * n_clean_windows, n_derivs)

            # 3. Iterate through protocols and visualize
            for i, protocol in enumerate(parametros_to_visualize):
                plot_title = f"Análise do Protocolo {i+1} para {filename_stem}"
                plot_filename = f"{filename_stem}_protocol_{i+1}.png"
                visualize_protocol_application(
                    eeg_data=eeg_data_for_analysis,
                    protocol_params=protocol,
                    fs=fs,
                    title=plot_title,
                    filename=plot_filename,
                    output_dir=Config.OUTPUT_DIR
                )

        except Exception as e:
            tqdm.write(f"\nERRO ao processar {os.path.basename(file_path)}: {e}")

if __name__ == '__main__':
    main()