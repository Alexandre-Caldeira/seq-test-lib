import numpy as np
from scipy.stats import chi2, f as finv
from scipy.optimize import minimize
from scipy.io import loadmat
import h5py
import matplotlib.pyplot as plt
import os
import warnings
from typing import Dict, List, Tuple
from tqdm import tqdm
from scipy.signal import butter, filtfilt, iirnotch
import glob

# ==============================================================================
# SECTION 0: UNIFIED CONFIGURATION BLOCK
# ==============================================================================
class Config:
    # --- I. Execution Control & Protocol Selection ---
    RUN_OPTIMIZATION = False # Set to True only to re-run the hours-long optimization
    PROTOCOLS_TO_VISUALIZE = [
        [200, 4, 240, 5, 0.05],
        [10, 10, 240, 8, 0.05],
        [230, 1, 240, 2, 0.05]
    ]

    # --- II. Data and Paths ---
    DATA_PATH = "C:\\Users\\alexa\\experimental_data\\todos\\ENTRADAS_PATRICIA"
    OUTPUT_DIR = "./protocol_visualizations_v2/"
    
    # --- III. NEW: Configurable Preprocessing Pipeline ---
    FILTER_TYPE = 'custom'  # Options: 'custom', 'Butterworth8', 'multi_BSPC2024', 'multi_MBEC2023', 'uni_BSPC2021', None
    ARTIFACT_REJECTION_TYPE = None # Options: 'percentile', 'legacy', 'uni_BSPC2021', None
    APPLY_NOTCH_FILTER = False
    APPLY_MEAN_REMOVAL = True
    REMOVE_FIRST_TWO_WINDOWS = True

    # --- IV. Parameters for Specific Preprocessing Methods ---
    CUSTOM_FILTER_PARAMS = {'low': 30.0, 'high': 300.0, 'order': 2}
    PERCENTILE_REMOVAL_AMOUNT = 5 # % of windows with highest peak amplitude to remove
    ARTIFACT_THRESHOLDS = {'uni_BSPC2021': 40e-6, 'legacy': 0.1 / 200}
    
    # --- V. Core Analysis Parameters ---
    TJ_DEFAULT = 32
    TARGET_ELECTRODES = ['F3', 'F4', 'Fz', 'T5', 'T6', 'Pz', 'Cz', 'Oz']
    AVAILABLE_ELECTROdes: List[str] = ['Fz', 'F3', 'F4', 'F7', 'Fcz', 'Cz', 'C3', 'C4', 'Pz', 'P3', 'P4', 'T3', 'T4', 'T5', 'T6', 'Oz']
    MAX_WINDOWS_PER_INTENSITY: Dict[str, int] = {'50dB': 240}
    SIGNAL_FREQUENCIES: List[int] = [81, 85, 89, 93, 83, 87, 91, 95]
    NOISE_FREQUENCIES: List[int] = [131, 137, 149, 151, 157, 163, 167, 173]

# --- Disabling Numba for Development ---
def njit(*args, **kwargs):
    def decorator(func): return func
    return decorator

# ==============================================================================
# SECTION 1: DERIVATION AND COHERENCE CALCULATION
# ==============================================================================
def dipolos(x: np.ndarray) -> np.ndarray:
    n_samples, n_channels = x.shape
    n_dipoles = n_channels + n_channels * (n_channels - 1) // 2
    y = np.zeros((n_samples, n_dipoles))
    y[:, :n_channels] = x
    idx = n_channels
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            y[:, idx] = x[:, i] - x[:, j]
            idx += 1
    return y

def msc_fft(Y: np.ndarray, M: int) -> np.ndarray:
    sum_Y = np.sum(Y, axis=1); ORD = np.zeros(Y.shape[0])
    denominator = M * np.sum(np.abs(Y)**2, axis=1)
    valid = denominator > 1e-12
    ORD[valid] = np.abs(sum_Y[valid])**2 / denominator[valid]
    return ORD

def _calculate_univariate_msc(data, tj, fs, alpha):
    M = data.shape[0] // tj
    if M <= 1: nf = tj // 2 + 1; return np.zeros(nf), np.arange(nf) * fs / tj, 1.0
    Y = np.fft.fft(data[:M*tj].reshape(M, tj), axis=1)[:, :tj//2+1]
    msc = msc_fft(Y.T, M)
    crit = 1 - alpha**(1/(M-1)) if M > 1 else 1.0
    return msc, np.arange(tj//2+1) * fs / tj, crit

def calculate_d_amsc(data, tj, fs, alpha=0.05):
    all_mscs = [_calculate_univariate_msc(data[:, i], tj, fs, alpha)[0] for i in range(data.shape[1])]
    F, crit = _calculate_univariate_msc(data[:, 0], tj, fs, alpha)[1:]
    return np.mean(all_mscs, axis=0), F, crit

def calculate_d_acsm(data, tj, fs, alpha=0.05):
    M, n_derivs = data.shape[0] // tj, data.shape[1]
    if M <= 1: nf = tj // 2 + 1; return np.zeros(nf), np.arange(nf) * fs / tj, 1.0
    all_csms = []
    for i in range(n_derivs):
        Y = np.fft.fft(data[:M*tj, i].reshape(M, tj), axis=1)[:, :tj//2+1]
        teta = np.angle(Y)
        sum_cos = np.sum(np.cos(teta), axis=0); sum_sin = np.sum(np.sin(teta), axis=0)
        all_csms.append((sum_cos**2 + sum_sin**2) / (M**2))
    crit = chi2.ppf(1 - alpha, 2) / (2 * M)
    return np.mean(all_csms, axis=0), np.arange(tj//2+1) * fs / tj, crit

# ==============================================================================
# SECTION 2: PROTOCOL LOGIC & PREPROCESSING
# ==============================================================================
def ETS(ord_values, MM_indices, alfa, NDC, vc_critical_values):
    NDC_ceil = int(np.ceil(NDC)); det = ord_values > vc_critical_values
    count, dr, time = 0, 0, MM_indices[-1]
    for ii in range(len(MM_indices)):
        if det[ii]: count += 1
        else: count = 0
        if count >= NDC_ceil: dr = 1; time = MM_indices[ii]; break
    return dr, time

def _monte_carlo_single_run(M_int, N_window, bin_idx):
    x = np.random.randn(N_window * M_int)
    fft_reshaped = np.fft.fft(x.reshape(N_window, M_int), axis=0)
    return msc_fft(fft_reshaped[bin_idx, :].reshape(1, -1).T, M_int)[0]

def VC_MSC(M_values, alfa, nRuns):
    VC_MC = np.array([np.quantile([_monte_carlo_single_run(int(M), 32, 8) for _ in range(nRuns)], 1.0-alfa) if M > 1 else 1.0 for M in M_values])
    return VC_MC, np.where(M_values > 1, 1 - alfa**(1.0 / (M_values - 1)), 1.0)

def preprocess_eeg_data(data, fs, config):
    processed_data = data.copy()
    
    # 1. Filtering
    filter_desc = "Filtro: Não"
    if config.FILTER_TYPE:
        params = {'order': 8}; stim_freq = 85.0
        if config.FILTER_TYPE == 'custom': params.update(config.CUSTOM_FILTER_PARAMS); low, high = params['low'], params['high']
        elif config.FILTER_TYPE == 'Butterworth8': low, high = 70.0, 110.0
        elif config.FILTER_TYPE == 'multi_BSPC2024': low, high = 1.0, 100.0
        elif config.FILTER_TYPE in ['multi_MBEC2023', 'uni_BSPC2021']: low, high = stim_freq - 1, stim_freq + 1
        b, a = butter(params['order'], [low, high], btype='bandpass', fs=fs)
        processed_data = filtfilt(b, a, processed_data, axis=0)
        filter_desc = f"Filtro: {config.FILTER_TYPE} ({low}-{high} Hz)"

    # 2. Notch Filter
    notch_desc = "Notch: Não"
    if config.APPLY_NOTCH_FILTER:
        b_n, a_n = iirnotch(60.0, 80, fs)
        processed_data = filtfilt(b_n, a_n, processed_data, axis=0)
        notch_desc = "Notch: Sim"

    # 3. Mean Removal
    if config.APPLY_MEAN_REMOVAL:
        processed_data -= np.mean(processed_data, axis=0, keepdims=True)
    
    # 4. Artifact Rejection
    num_windows_before = processed_data.shape[1]
    windows_to_keep = np.ones(num_windows_before, dtype=bool)
    artifact_desc = "Rej. Artefato: Não"
    if config.ARTIFACT_REJECTION_TYPE:
        reshaped = processed_data.reshape(config.TJ_DEFAULT, num_windows_before, -1)
        peak_amps = np.max(np.abs(reshaped), axis=0) # Shape (windows, derivations)
        peak_per_window = np.max(peak_amps, axis=1) # Shape (windows,)

        if config.ARTIFACT_REJECTION_TYPE in config.ARTIFACT_THRESHOLDS:
            threshold = config.ARTIFACT_THRESHOLDS[config.ARTIFACT_REJECTION_TYPE]
            windows_to_keep = peak_per_window < threshold
            artifact_desc = f"Rej. Artefato: {config.ARTIFACT_REJECTION_TYPE} (<{threshold*1e6:.0f} uV)"
        elif config.ARTIFACT_REJECTION_TYPE == 'percentile':
            if peak_per_window.size > 0:
                threshold = np.percentile(peak_per_window, 100 - config.PERCENTILE_REMOVAL_AMOUNT)
                windows_to_keep = peak_per_window <= threshold
                artifact_desc = f"Rej. Artefato: Percentil (Top {config.PERCENTILE_REMOVAL_AMOUNT}%)"
    
    processed_data = processed_data.reshape(config.TJ_DEFAULT, num_windows_before, -1)[:, windows_to_keep, :].reshape(-1, data.shape[-1])
    num_windows_after = processed_data.shape[0] // config.TJ_DEFAULT if config.TJ_DEFAULT > 0 else 0
    
    # 5. Remove first windows
    if config.REMOVE_FIRST_TWO_WINDOWS and num_windows_after > 2:
        processed_data = processed_data[config.TJ_DEFAULT*2:, :]
        num_windows_after -= 2

    subtitle = f"{filter_desc} | {notch_desc} | {artifact_desc} | Janelas: {num_windows_after}/{num_windows_before}"
    return processed_data, subtitle

# ==============================================================================
# SECTION 3: VISUALIZATION
# ==============================================================================

def visualize_protocol_application(eeg_proc, eeg_raw, protocol_params, fs, title, subtitle, filename, output_dir):
    Mmin, Mstep, Mmax, NDC, alpha = protocol_params; tj = Config.TJ_DEFAULT
    num_samples_proc, num_derivs = eeg_proc.shape
    num_windows_proc = num_samples_proc // tj
    if num_windows_proc < Mmin:
        tqdm.write(f"AVISO: Pulando plot '{filename}', Mmin ({Mmin}) > janelas limpas ({num_windows_proc}).")
        return
    
    MM_indices = np.arange(int(Mmin), int(Mmax) + 1, int(Mstep))
    MM_indices = MM_indices[MM_indices <= num_windows_proc]
    if MM_indices.size < 2: return

    xfft_proc = np.fft.fft(eeg_proc[:num_windows_proc*tj].reshape(num_windows_proc, tj, num_derivs), axis=1)
    ord_values = np.array([np.mean([msc_fft(xfft_proc[:M, :, d].T, M)[8] for d in range(num_derivs)]) for M in MM_indices])
    _, vc_teorico = VC_MSC(MM_indices, alpha, nRuns=1000)
    det_res, det_time = ETS(ord_values, MM_indices, alpha, NDC, vc_teorico)

    d_amsc, F, crit_d_amsc = calculate_d_amsc(eeg_proc, tj, fs, alpha)
    d_acsm, _, crit_d_acsm = calculate_d_acsm(eeg_proc, tj, fs, alpha)

    fig, axs = plt.subplots(4, 1, figsize=(20, 25), gridspec_kw={'height_ratios': [1.5, 1, 1, 1]})
    fig.suptitle(title, fontsize=20)
    fig.text(0.5, 0.93, subtitle, ha='center', fontsize=14)

    # Plot 1: Raw vs Processed Signal Example
    time_vec = np.arange(eeg_raw.shape[0]) / fs
    axs[0].plot(time_vec, eeg_raw[:, 0] * 1e6, 'r-', alpha=0.6, label=f'Original (Derivação 1)')
    if eeg_proc.size > 0:
        time_vec_proc = np.arange(eeg_proc.shape[0]) / fs
        axs[0].plot(time_vec_proc, eeg_proc[:, 0] * 1e6, 'b-', alpha=0.8, label=f'Processado (Derivação 1)')
    axs[0].set_title("Sinal Exemplo: Original vs. Pós-Processamento"); axs[0].set_xlabel("Tempo (s)"); axs[0].set_ylabel("Amplitude (uV)"); axs[0].legend(); axs[0].grid(True, ls='--');

    # Plot 2: Protocol Decision
    status = f"DETECTADO em M={int(det_time)}" if det_res else "NÃO DETECTADO"
    axs[1].plot(MM_indices, ord_values, 'o-', c='blue', label='MSC Médio (bin 8)')
    axs[1].plot(MM_indices, vc_teorico, '--', c='red', label='Valor Crítico')
    axs[1].set_title(f"Decisão do Protocolo {protocol_params} -> {status}"); axs[1].set_xlabel("Número de Janelas (M)"); axs[1].set_ylabel("MSC"); axs[1].legend(); axs[1].grid(True, ls='--')

    # Plots 3 & 4: Coherence Spectrums
    for ax, data, crit, name in [(axs[2], d_amsc, crit_d_amsc, "D-aMSC"), (axs[3], d_acsm, crit_d_acsm, "D-aCSM")]:
        ax.plot(F, data, label=name); ax.axhline(crit, c='k', ls='--', label='Limiar')
        ax.set_title(f"Espectro de Coerência ({name})"); ax.set_xlabel("Frequência (Hz)"); ax.set_ylabel("Coerência"); ax.legend(); ax.grid(True, ls='--'); ax.set_xlim(0, 200)
        for freq in Config.SIGNAL_FREQUENCIES: ax.axvline(x=freq, color='g', linestyle=':', alpha=0.7)
        for freq in Config.NOISE_FREQUENCIES: ax.axvline(x=freq, color='r', linestyle=':', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.92]); plt.savefig(os.path.join(output_dir, filename)); plt.close(fig)

# ==============================================================================
# SECTION 4: MAIN ANALYSIS BLOCK
# ==============================================================================

def load_mat_file(filepath: str) -> Tuple[np.ndarray, float]:
    try: return loadmat(filepath)['x'], loadmat(filepath)['Fs'].flatten()[0]
    except NotImplementedError:
        tqdm.write(f"INFO: '{os.path.basename(filepath)}' é v7.3, usando h5py.")
        with h5py.File(filepath, 'r') as f: return f['x'][:].transpose(1, 2, 0), f['Fs'][0, 0]

def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    if Config.RUN_OPTIMIZATION: print("FATAL: Otimização desativada."); return

    all_mat_files = glob.glob(os.path.join(Config.DATA_PATH, '*.mat'))
    electrode_indices = [Config.AVAILABLE_ELECTROdes.index(e) for e in Config.TARGET_ELECTRODES]

    for file_path in tqdm(all_mat_files, desc="Processando Voluntários"):
        try:
            filename_stem = os.path.splitext(os.path.basename(file_path))[0]
            if not filename_stem.endswith("50dB"): continue
            
            raw_data, fs = load_mat_file(file_path)
            x_subset = raw_data[:, :, electrode_indices]
            
            # Create all derivations
            all_derivations = np.stack([dipolos(x_subset[:, i, :]) for i in range(x_subset.shape[1])], axis=1)
            num_samples, num_windows, num_derivs = all_derivations.shape
            eeg_raw_continuous = all_derivations.transpose(0,2,1).reshape(num_samples, num_derivs, num_windows).reshape(num_samples*num_windows, num_derivs, order='F')

            # Apply configurable preprocessing
            eeg_proc_continuous, subtitle = preprocess_eeg_data(eeg_raw_continuous, fs, Config)

            for i, protocol in enumerate(Config.PROTOCOLS_TO_VISUALIZE):
                visualize_protocol_application(
                    eeg_proc=eeg_proc_continuous, eeg_raw=eeg_raw_continuous, 
                    protocol_params=protocol, fs=fs, title=f"Análise para {filename_stem} | Protocolo {i+1}",
                    subtitle=subtitle, filename=f"{filename_stem}_protocol_{i+1}.png", output_dir=Config.OUTPUT_DIR
                )
        except Exception as e:
            tqdm.write(f"\nERRO CRÍTICO ao processar {os.path.basename(file_path)}: {e}")

if __name__ == '__main__':
    main()