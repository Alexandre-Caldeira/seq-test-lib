import numpy as np
from scipy.stats import chi2, f as finv
from scipy.optimize import minimize
from scipy.io import loadmat
import h5py
import yaml
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import os
import traceback
from typing import Dict, List, Tuple
from tqdm import tqdm
from scipy.signal import butter, filtfilt, iirnotch
import glob
from functools import partial

# ==============================================================================
# SECTION 0: CONFIGURATION MANAGEMENT
# ==============================================================================

DEFAULT_CONFIG = {
    # --- I. Execution Control & Protocol Selection ---
    'APPLY_PREPROCESSING_TO_SIMULATIONS': True,
    'PROTOCOLS_TO_VISUALIZE': [
        [200, 4, 240, 5, 0.05], [10, 10, 240, 8, 0.05], [230, 1, 240, 2, 0.05]
    ],
    # --- II. Data and Paths ---
    'DATA_PATH': "C:\\Users\\alexa\\experimental_data\\todos\\ENTRADAS_PATRICIA",
    'BASE_OUTPUT_DIR': "../protocol_visualizations/",
    # --- III. Configurable Preprocessing Pipeline ---
    'FILTER_TYPE': 'custom',
    'ARTIFACT_REJECTION_TYPE': None,
    'APPLY_NOTCH_FILTER': False,
    'APPLY_MEAN_REMOVAL': True,
    'REMOVE_FIRST_TWO_WINDOWS': True,
    # --- IV. Parameters for Specific Preprocessing Methods ---
    'CUSTOM_FILTER_PARAMS': {'low': 30.0, 'high': 300.0, 'order': 2},
    'PERCENTILE_REMOVAL_AMOUNT': 10,
    'ARTIFACT_THRESHOLDS': {'uni_BSPC2021': 40e-6, 'legacy': 0.1 / 200},
    # --- V. Core Analysis Parameters ---
    'TARGET_ELECTRODES': 'Fz',
    'AVAILABLE_ELECTRODES': ['Fz','F3','F4','F7','Fcz','Cz','C3','C4','Pz','P3','P4','T3','T4','T5','T6','Oz'],
    'SIGNAL_FREQUENCIES': [81, 85, 89, 93, 83, 87, 91, 95],
    'NOISE_FREQUENCIES': [131, 137, 149, 151, 157, 163, 167, 173]
}

def load_config(args):
    if args.config and os.path.exists(args.config):
        print(f"Carregando configuração de: '{args.config}'")
        with open(args.config, 'r') as f: user_config = yaml.safe_load(f)
        config = DEFAULT_CONFIG.copy(); config.update(user_config)
        return config
    print("Nenhum arquivo de configuração especificado. Usando configuração padrão.")
    return DEFAULT_CONFIG

def save_config(config, output_dir):
    filepath = os.path.join(output_dir, "config.yml")
    with open(filepath, 'w') as f: yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=4)
    print(f"Configuração da execução salva em: '{filepath}'\n")

# ==============================================================================
# SECTION 1: DERIVATION AND COHERENCE CALCULATION
# ==============================================================================

def dipolos(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1: x = x[:, np.newaxis]
    n_samples, n_channels = x.shape
    n_dipoles = n_channels + n_channels * (n_channels - 1) // 2
    y = np.zeros((n_samples, n_dipoles)); y[:, :n_channels] = x
    idx = n_channels
    for i in range(n_channels):
        for j in range(i + 1, n_channels): y[:, idx] = x[:, i] - x[:, j]; idx += 1
    return y

def msc_fft(Y: np.ndarray, M: int) -> np.ndarray:
    sum_Y = np.sum(Y, axis=1); ORD = np.zeros(Y.shape[0])
    denominator = M * np.sum(np.abs(Y)**2, axis=1)
    valid = denominator > 1e-12
    ORD[valid] = np.abs(sum_Y[valid])**2 / denominator[valid]
    return ORD

def _calculate_univariate_msc(data, tj, fs, alpha):
    M = data.shape[0] // tj
    if M <= 1: nf = tj//2+1; return np.zeros(nf), np.arange(nf)*fs/tj, 1.0
    Y = np.fft.fft(data[:M*tj].reshape(M, tj), axis=1)[:, :tj//2+1]
    msc = msc_fft(Y.T, M)
    crit = 1 - alpha**(1/(M-1)) if M > 1 else 1.0
    return msc, np.arange(tj//2+1)*fs/tj, crit

def calculate_empirical_threshold(metric_func_core, n_sims, M, n_derivs, tj, fs, alpha):
    metric_results = np.zeros(n_sims)
    for i in range(n_sims):
        noise = np.random.randn(M * tj, n_derivs)
        metric_spectrum, _, _ = metric_func_core(noise, tj, fs, alpha)
        metric_results[i] = np.max(metric_spectrum)
    return np.quantile(metric_results, 1 - alpha)

def _calculate_d_amsc_core(data, tj, fs, alpha, config=None, filter_coeffs=None):
    """Pure calculation function, without thresholding logic."""
    if filter_coeffs: data = filtfilt(filter_coeffs[0], filter_coeffs[1], data, axis=0)
    all_mscs = [_calculate_univariate_msc(data[:, i], tj, fs, alpha)[0] for i in range(data.shape[1])]
    F, _ = _calculate_univariate_msc(data[:, 0], tj, fs, alpha)[1:]
    return np.mean(all_mscs, axis=0), F, None # Return None for crit

def calculate_d_amsc(data, tj, fs, alpha, config, filter_coeffs):
    """Orchestrator: calculates spectrum and the correct threshold."""
    spectrum, F, _ = _calculate_d_amsc_core(data, tj, fs, alpha)
    if config['APPLY_PREPROCESSING_TO_SIMULATIONS']:
        M, n_derivs = data.shape[0] // tj, data.shape[1]
        core_func = partial(_calculate_d_amsc_core, config=config, filter_coeffs=filter_coeffs)
        crit = calculate_empirical_threshold(core_func, 10000, M, n_derivs, tj, fs, alpha)
    else:
        M = data.shape[0] // tj; crit = 1 - alpha**(1/(M-1)) if M > 1 else 1.0
    return spectrum, F, crit

def _calculate_d_acsm_core(data, tj, fs, alpha, config=None, filter_coeffs=None):
    """Pure calculation function for D-aCSM."""
    if filter_coeffs: data = filtfilt(filter_coeffs[0], filter_coeffs[1], data, axis=0)
    M, n_derivs = data.shape[0]//tj, data.shape[1]
    if M <= 1: nf = tj//2+1; return np.zeros(nf), np.arange(nf)*fs/tj, None
    all_csms = []
    for i in range(n_derivs):
        Y = np.fft.fft(data[:M*tj, i].reshape(M, tj), axis=1)[:, :tj//2+1]
        teta = np.angle(Y)
        sum_cos = np.sum(np.cos(teta), axis=0); sum_sin = np.sum(np.sin(teta), axis=0)
        all_csms.append((sum_cos**2 + sum_sin**2) / (M**2))
    F = np.arange(tj//2+1)*fs/tj
    return np.mean(all_csms, axis=0), F, None

def calculate_d_acsm(data, tj, fs, alpha, config, filter_coeffs):
    """Orchestrator for D-aCSM spectrum and threshold."""
    spectrum, F, _ = _calculate_d_acsm_core(data, tj, fs, alpha)
    if config['APPLY_PREPROCESSING_TO_SIMULATIONS']:
        M, n_derivs = data.shape[0] // tj, data.shape[1]
        core_func = partial(_calculate_d_acsm_core, config=config, filter_coeffs=filter_coeffs)
        crit = calculate_empirical_threshold(core_func, 200, M, n_derivs, tj, fs, alpha)
    else:
        M = data.shape[0] // tj; crit = chi2.ppf(1 - alpha, 2) / (2 * M) if M > 0 else 1.0
    return spectrum, F, crit

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

def _monte_carlo_single_run(M_int, N_window, bin_idx, filter_coeffs):
    x = np.random.randn(N_window * M_int)
    if filter_coeffs: x = filtfilt(filter_coeffs[0], filter_coeffs[1], x)
    fft_reshaped = np.fft.fft(x.reshape(N_window, M_int), axis=0)
    return msc_fft(fft_reshaped[bin_idx, :].reshape(1, -1).T, M_int)[0]

def VC_MSC(M_values, alfa, nRuns, filter_coeffs):
    task = partial(_monte_carlo_single_run, N_window=32, bin_idx=8, filter_coeffs=filter_coeffs)
    VC_MC = np.array([np.quantile([task(M_int=int(M)) for _ in range(nRuns)], 1.0-alfa) if M > 1 else 1.0 for M in M_values])
    return VC_MC, np.where(M_values > 1, 1 - alfa**(1.0 / (M_values - 1)), 1.0)

def preprocess_eeg_data(data_continuous, fs, tj, config, filter_coeffs):
    processed_data = data_continuous.copy()
    
    filter_desc = "Filtro: Não"
    if config['FILTER_TYPE'] and filter_coeffs:
        processed_data = filtfilt(filter_coeffs[0], filter_coeffs[1], processed_data, axis=0)
        params = config['CUSTOM_FILTER_PARAMS']
        filter_desc = f"Filtro: {config['FILTER_TYPE']} ({params['low']}-{params['high']} Hz)"

    notch_desc = "Notch: Não"
    if config['APPLY_NOTCH_FILTER']:
        b_n, a_n = iirnotch(60.0, 80, fs); processed_data = filtfilt(b_n, a_n, processed_data, axis=0); notch_desc = "Notch: Sim"

    if config['APPLY_MEAN_REMOVAL']: processed_data -= np.mean(processed_data, axis=0, keepdims=True)
    
    total_samples = processed_data.shape[0]
    samples_to_keep = (total_samples // tj) * tj
    processed_data = processed_data[:samples_to_keep, :]
    
    num_windows_before = processed_data.shape[0] // tj
    artifact_desc = "Rej. Artefato: Não"
    
    if config['ARTIFACT_REJECTION_TYPE'] and num_windows_before > 0:
        reshaped = processed_data.reshape(num_windows_before, tj, processed_data.shape[1])
        peak_per_window = np.max(np.abs(reshaped), axis=(1, 2))
        
        if config['ARTIFACT_REJECTION_TYPE'] in config['ARTIFACT_THRESHOLDS']:
            threshold = config['ARTIFACT_THRESHOLDS'][config['ARTIFACT_REJECTION_TYPE']]
            windows_to_keep = peak_per_window < threshold
            artifact_desc = f"Rej. Artefato: {config['ARTIFACT_REJECTION_TYPE']} (<{threshold*1e6:.0f} uV)"
        elif config['ARTIFACT_REJECTION_TYPE'] == 'percentile':
            threshold = np.percentile(peak_per_window, 100 - config['PERCENTILE_REMOVAL_AMOUNT'])
            windows_to_keep = peak_per_window <= threshold
            artifact_desc = f"Rej. Artefato: Percentil (Top {config['PERCENTILE_REMOVAL_AMOUNT']}%)"
        
        processed_data = reshaped[windows_to_keep, :, :].reshape(-1, data_continuous.shape[1])

    num_windows_after = processed_data.shape[0] // tj
    if config['REMOVE_FIRST_TWO_WINDOWS'] and num_windows_after > 2:
        processed_data = processed_data[tj*2:, :]; num_windows_after -= 2

    subtitle = f"{filter_desc} | {notch_desc} | {artifact_desc} | Janelas: {num_windows_after}/{num_windows_before}"
    return processed_data, subtitle

# ==============================================================================
# SECTION 3: VISUALIZATION
# ==============================================================================

def visualize_protocol_application(eeg_proc, eeg_raw, protocol_params, fs, tj, title, subtitle, filename, output_dir, config, filter_coeffs):
    Mmin, Mstep, Mmax, NDC, alpha = protocol_params
    num_samples_proc, num_derivs = eeg_proc.shape
    num_windows_proc = num_samples_proc // tj
    if num_windows_proc < Mmin: return
    
    MM_indices = np.arange(int(Mmin), int(Mmax) + 1, int(Mstep))
    MM_indices = MM_indices[MM_indices <= num_windows_proc]
    if MM_indices.size < 2: return

    xfft_proc = np.fft.fft(eeg_proc[:num_windows_proc*tj].reshape(num_windows_proc, tj, num_derivs), axis=1)
    ord_values = np.array([np.mean([msc_fft(xfft_proc[:M, :, d].T, M)[8] for d in range(num_derivs)]) for M in MM_indices])
    
    vc_mc, vc_theoretical = VC_MSC(MM_indices, alpha, 1000, filter_coeffs if config['APPLY_PREPROCESSING_TO_SIMULATIONS'] else None)
    vc_to_use = vc_mc if config['APPLY_PREPROCESSING_TO_SIMULATIONS'] else vc_theoretical
    vc_label = "Valor Crítico (Empírico)" if config['APPLY_PREPROCESSING_TO_SIMULATIONS'] else "Valor Crítico (Teórico)"
    
    det_res, det_time = ETS(ord_values, MM_indices, alpha, NDC, vc_to_use)

    d_amsc, F, crit_d_amsc = calculate_d_amsc(eeg_proc, tj, fs, alpha, config, filter_coeffs)
    d_acsm, _, crit_d_acsm = calculate_d_acsm(eeg_proc, tj, fs, alpha, config, filter_coeffs)

    fig, axs = plt.subplots(5, 1, figsize=(20, 28), gridspec_kw={'height_ratios': [1, 1, 1, 1, 1]})
    fig.suptitle(title, fontsize=20); fig.text(0.5, 0.96, subtitle, ha='center', fontsize=14)

    axs[0].plot(np.arange(eeg_raw.shape[0]) / fs, eeg_raw[:, 0] * 1e6, 'r-', alpha=0.8, label='Original (Derivação 1)'); axs[0].set_title("Sinal Exemplo: Original"); axs[0].legend(); axs[0].grid(True, ls='--')
    if eeg_proc.size > 0: axs[1].plot(np.arange(eeg_proc.shape[0]) / fs, eeg_proc[:, 0] * 1e6, 'b-', alpha=0.8, label='Processado (Derivação 1)')
    axs[1].set_title("Sinal Exemplo: Pós-Processamento"); axs[1].legend(); axs[1].grid(True, ls='--')
    
    status = f"DETECTADO em M={int(det_time)}" if det_res else "NÃO DETECTADO"
    axs[2].plot(MM_indices, ord_values, 'o-', c='blue', label='MSC Médio (bin 8)'); axs[2].plot(MM_indices, vc_to_use, '--', c='red', label=vc_label)
    axs[2].set_title(f"Decisão do Protocolo {protocol_params} -> {status}"); axs[2].legend(); axs[2].grid(True, ls='--')

    crit_label = "Limiar Empírico" if config['APPLY_PREPROCESSING_TO_SIMULATIONS'] else "Limiar Teórico"
    for ax, data, crit, name in [(axs[3], d_amsc, crit_d_amsc, "D-aMSC"), (axs[4], d_acsm, crit_d_acsm, "D-aCSM")]:
        ax.plot(F, data, label=name); ax.axhline(crit, c='k', ls='--', label=crit_label)
        ax.set_title(f"Espectro de Coerência ({name})"); ax.legend(); ax.grid(True, ls='--'); ax.set_xlim(0, 500)
        for freq in config['SIGNAL_FREQUENCIES']: ax.axvline(x=freq, color='g', linestyle=':', alpha=0.7)
        for freq in config['NOISE_FREQUENCIES']: ax.axvline(x=freq, color='r', linestyle=':', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(os.path.join(output_dir, filename)); plt.close(fig)

# ==============================================================================
# SECTION 4: MAIN ANALYSIS BLOCK
# ==============================================================================

def load_mat_file(filepath: str) -> Tuple[np.ndarray, float]:
    try: return loadmat(filepath)['x'], loadmat(filepath)['Fs'].flatten()[0]
    except NotImplementedError:
        with h5py.File(filepath, 'r') as f: return f['x'][:].transpose(1, 2, 0), f['Fs'][0, 0]

def main(config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(config['BASE_OUTPUT_DIR'], f"run_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)
    save_config(config, run_output_dir)
    
    if isinstance(config['TARGET_ELECTRODES'], str):
        config['TARGET_ELECTRODES'] = [config['TARGET_ELECTRODES']]

    all_mat_files = glob.glob(os.path.join(config['DATA_PATH'], '*.mat'))
    electrode_indices = [config['AVAILABLE_ELECTRODES'].index(e) for e in config['TARGET_ELECTRODES']]
    
    filter_coeffs = None
    if config['FILTER_TYPE']:
        fs_placeholder = 1000
        params = {'order': 8}
        if config['FILTER_TYPE'] == 'custom': params.update(config['CUSTOM_FILTER_PARAMS']); low, high = params['low'], params['high']
        else: low, high = 1.0, 100.0
        filter_coeffs = butter(params['order'], [low, high], btype='bandpass', fs=fs_placeholder)

    for file_path in tqdm(all_mat_files, desc="Processando Voluntários"):
        try:
            filename_stem = os.path.splitext(os.path.basename(file_path))[0]
            if not filename_stem.endswith("50dB"): continue
            
            raw_data, fs = load_mat_file(file_path)
            tj = raw_data.shape[0]
            
            x_subset = raw_data[:, :, electrode_indices]
            all_derivations = np.stack([dipolos(x_subset[:, i, :]) for i in range(x_subset.shape[1])], axis=1)
            
            num_samples, num_windows, num_derivs = all_derivations.shape
            eeg_raw_continuous = all_derivations.transpose(1, 0, 2).reshape(num_windows * num_samples, num_derivs)

            eeg_proc_continuous, subtitle = preprocess_eeg_data(eeg_raw_continuous, fs, tj, config, filter_coeffs)

            if eeg_proc_continuous.shape[0] == 0: continue

            for i, protocol in enumerate(config['PROTOCOLS_TO_VISUALIZE']):
                visualize_protocol_application(
                    eeg_proc=eeg_proc_continuous, eeg_raw=eeg_raw_continuous, 
                    protocol_params=protocol, fs=fs, tj=tj, title=f"Análise para {filename_stem} | Protocolo {i+1}",
                    subtitle=subtitle, filename=f"{filename_stem}_protocol_{i+1}.png", output_dir=run_output_dir,
                    config=config, filter_coeffs=filter_coeffs
                )
        except Exception as e:
            print(f"\n--- ERRO CRÍTICO AO PROCESSAR O ARQUIVO: {os.path.basename(file_path)} ---")
            print(f"CONDIÇÃO DO ERRO: Parâmetros -> FILTER='{config['FILTER_TYPE']}', ARTIFACTS='{config['ARTIFACT_REJECTION_TYPE']}'")
            print(f"MENSAGEM: {e}")
            traceback.print_exc()
            print("--- FIM DO RELATÓRIO DE ERRO ---\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Executa o pipeline de análise de coerência de EEG.")
    parser.add_argument('--config', type=str, help='Caminho para o arquivo de configuração .yml.')
    args = parser.parse_args()
    
    final_config = load_config(args)
    main(final_config)