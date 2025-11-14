20251107_184500_v2

Based on code A, add methods to code B and C in order to draw a Pareto plot comparing D-aMSC, D-aCSM, MMSC and MCSM as implemented on code A. Use the annex codes as original implementation reference (legacy code in MATLAB). Our desired plots are TPR (%) x Exam duration (s) and FPR (%) x Exam duration (s) similarly to the ones in annex.

===
CODE A:

-- coding: utf-8 --

"""
CÉLULA DE PROCESSAMENTO EXPERIMENTAL CONFIGURÁVEL (v24.6.3 - Correção MMSC/MCSM)

Objetivo: Um pipeline de análise e visualização de EEG altamente configurável,
com gestão de parâmetros e saídas organizadas para garantir a reprodutibilidade.

NOVAS FUNCIONALIONALIDADES E CORREÇÕES:

CORREÇÃO ALGORÍTMICA (MMSC/MCSM): As funções calculate_mmsc e calculate_mcsm
foram corrigidas para corresponderem exatamente às implementações de referência
em MATLAB.

calculate_mmsc: Corrigido o valor do elemento da matriz aumentada e a
fórmula final para (1 - real(...))/M.

calculate_mcsm: Corrigida a ordem das operações para primeiro mediar sobre
os canais e depois somar sobre as épocas, e corrigido o denominador.
Essas correções resolvem o problema de "desempenho inesperado".

CLAREZA NA SELEÇÃO DE MÉTRICAS: Adicionados comentários para explicar que a
geração de gráficos MMSC/MCSM é intencional para os modos 'dipole' e 'channel',
enquanto D-aMSC/D-aCSM são para o modo 'all_derivations'.
"""
import os
import glob
import yaml
import argparse
from datetime import datetime
import numpy as np
from tqdm import tqdm
from scipy.signal import butter, filtfilt, iirnotch
from scipy.stats import f as finv, chi2
import matplotlib.pyplot as plt
import h5py
import itertools
import math

==============================================================================
GESTÃO DE CONFIGURAÇÃO
==============================================================================

DEFAULT_CONFIG = {
# --- 1. Parâmetros Gerais e de Dados ---
'EXPERIMENTAL_DATA_DIR': 'C:/Users/alexa/experimental_data/todos/Sinais_EEG/',
'BASE_OUTPUT_DIR': './plots/coherence_analysis/',

code
Code
download
content_copy
expand_less
# --- 2. Controles de Execução ---
'RUN_COHERENCE_PLOTS': True,
'SKIP_EXISTING_PLOTS': True,

# --- 3. Seleção de Canal e Modo de Análise ---
'ANALYSIS_MODE': 'channel',
'CHANNELS_TO_USE': ['Fz', 'Fcz', 'Cz', 'Pz','T3', 'T4','Oz'],

# --- 4. Pipeline de Pré-processamento ---
'FILTER_TYPE': 'multi_BSPC2024',
'ARTIFACT_REJECTION_TYPE': None,
'APPLY_NOTCH_FILTER': None,
'APPLY_MEAN_REMOVAL': True,
'REMOVE_FIRST_TWO_WINDOWS': False,

# --- 5. Parâmetros para Opções Específicas ---
'CUSTOM_FILTER_LOW_CUT': 30.0,
'CUSTOM_FILTER_HIGH_CUT': 300.0,
'CUSTOM_FILTER_ORDER': 2,
'PERCENTILE_REMOVAL_AMOUNT': 5,

# --- 6. Frequências de Análise ---
'SIGNAL_FREQUENCIES': [81, 85, 89, 93, 83, 87, 91, 95],
'NOISE_FREQUENCIES': [131, 137, 149, 151, 157, 163, 167, 173],

# --- 7. Parâmetros Fixos ---
'NOTCH_FREQ': 60.0,
'NOTCH_Q': 100

}

def load_config_from_yaml(config_file):
print(f"Carregando configuração de: '{config_file}'")
with open(config_file, 'r') as f:
loaded_config = yaml.safe_load(f)
config = DEFAULT_CONFIG.copy(); config.update(loaded_config)
return config

def save_config_to_yaml(config, run_output_dir):
filepath = os.path.join(run_output_dir, "config.yml")
with open(filepath, 'w') as f:
yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=4)
print(f"Configuração da execução salva em: '{filepath}'\n")

==============================================================================
FUNÇÕES DE CÁLCULO E HELPERS
==============================================================================

def msweep(matrix, r):
A = np.copy(matrix).astype(np.complex128)
for k in range(r):
d = A[k, k]
if np.abs(d) < 1e-12: d = 1e-12
col_k, row_k = A[:, k].copy(), A[k, :].copy()
A -= np.outer(col_k, row_k) / d; A[k, :] = row_k / d; A[:, k] = -col_k / d; A[k, k] = 1.0 / d
return A

def _calculate_univariate_msc(data, tj, fs, alpha):
M = data.shape[0] // tj
if M <= 1:
nf = tj // 2 + 1
return np.zeros(nf), np.arange(nf) * fs / tj, 1.0
epochs = data.flatten()[:M*tj].reshape(M, tj)
nf = tj // 2 + 1
Y = np.fft.fft(epochs, axis=1)[:, :nf]
sum_Y = np.sum(Y, axis=0)
sum_Y_sq_mag = np.sum(np.abs(Y)**2, axis=0)
msc_values = np.zeros(nf)
valid_idx = sum_Y_sq_mag > 1e-12
msc_values[valid_idx] = np.abs(sum_Y[valid_idx])2 / (M * sum_Y_sq_mag[valid_idx])
crit = 1 - alpha(1/(M-1)) if M > 1 else 1.0
F = np.arange(nf) * fs / tj
return msc_values, F, crit

def _calculate_univariate_csm(data, tj, fs, alpha):
M = data.shape[0] // tj
if M <= 1:
nf = tj // 2 + 1
return np.zeros(nf), np.arange(nf) * fs / tj, 1.0
epochs = data.flatten()[:M * tj].reshape(M, tj)
nf = tj // 2 + 1
Y = np.fft.fft(epochs, axis=1)[:, :nf]
teta = np.angle(Y)
sum_cos = np.sum(np.cos(teta), axis=0)
sum_sin = np.sum(np.sin(teta), axis=0)
csm_values = (sum_cos2 + sum_sin2) / (M**2)
crit = chi2.ppf(1 - alpha, 2) / (2 * M) if M > 0 else 1.0
F = np.arange(nf) * fs / tj
return csm_values, F, crit

def calculate_mmsc(data, tj, fs, alpha=0.05):
"""
Calcula MMSC (multivariado) ou MSC (univariado).
VERSÃO CORRIGIDA (v24.6.3) para corresponder à implementação de referência do MATLAB.
"""
n_samples, n_channels = data.shape if data.ndim > 1 else (data.shape[0], 1)
if n_channels == 1:
return _calculate_univariate_msc(data, tj, fs, alpha)

code
Code
download
content_copy
expand_less
M = n_samples // tj
nf = tj // 2 + 1
F = np.arange(nf) * fs / tj
if M <= n_channels: return np.zeros(nf), F, 1.0

epochs = data[:M*tj, :].T.reshape(n_channels, M, tj)
Sfft = np.fft.fft(epochs, axis=2)[:, :, :nf]
mmsc_values = np.zeros(nf)

for kf in range(nf):
    Sfft_slice = Sfft[:, :, kf]
    spec_matrix_a = np.zeros((n_channels + 1, n_channels + 1), dtype=np.complex128)
    
    # Monta a matriz de espectro cruzado (bloco superior esquerdo)
    spec_matrix_a[:n_channels, :n_channels] = Sfft_slice @ Sfft_slice.conj().T
    
    # Vetor V (soma dos espectros sobre as épocas)
    V = np.sum(Sfft_slice, axis=1)
    
    # Preenche a matriz aumentada
    spec_matrix_a[n_channels, :n_channels] = V.conj()
    spec_matrix_a[:n_channels, n_channels] = V
    
    # --- CORREÇÃO 1: O valor deste elemento deve ser 1, não M ---
    spec_matrix_a[n_channels, n_channels] = 1
    
    # Aplica o operador sweep
    spec_matrix_as = msweep(spec_matrix_a, n_channels)
    
    # --- CORREÇÃO 2: A fórmula final deve ser (1 - real(...))/M ---
    mmsc_values[kf] = (1 - np.real(spec_matrix_as[n_channels, n_channels])) / M

Fcrit = finv.ppf(1 - alpha, 2 * n_channels, 2 * (M - n_channels))
k2Ncrit = (n_channels / (M - n_channels)) * Fcrit / (1 + (n_channels / (M - n_channels)) * Fcrit)
return mmsc_values, F, k2Ncrit

def calculate_mcsm(data, tj, fs, alpha=0.05):
"""
Calcula MCSM (multivariado) ou CSM (univariado).
VERSÃO CORRIGIDA (v24.6.3) para corresponder à implementação de referência do MATLAB.
"""
n_samples, n_channels = data.shape if data.ndim > 1 else (data.shape[0], 1)
if n_channels == 1:
return _calculate_univariate_csm(data, tj, fs, alpha)

code
Code
download
content_copy
expand_less
M = n_samples // tj
nf = tj // 2 + 1
F = np.arange(nf) * fs / tj
if M == 0: return np.zeros(nf), F, 1.0

epochs = data[:M * tj, :].reshape(M, tj, n_channels)
Y = np.fft.fft(epochs, axis=1)[:, :nf, :] # Shape: (M, nf, n_channels)

# --- LÓGICA DE CÁLCULO CORRIGIDA ---
# 1. Obter fase
teta = np.angle(Y)

# 2. Calcular a média de cos e sin SOBRE OS CANAIS (axis=2)
C_mean_over_channels = np.mean(np.cos(teta), axis=2)
S_mean_over_channels = np.mean(np.sin(teta), axis=2)

# 3. Calcular o ângulo de fase médio (para cada época e frequência)
teta_med = np.arctan2(S_mean_over_channels, C_mean_over_channels) # Shape: (M, nf)

# 4. Somar os cos e sin do ângulo médio SOBRE AS ÉPOCAS (axis=0)
sum_cos = np.sum(np.cos(teta_med), axis=0)
sum_sin = np.sum(np.sin(teta_med), axis=0)

# 5. Calcular o valor final de MCSM com o denominador correto
mcsm_values = (sum_cos**2 + sum_sin**2) / (M**2)

csmNcrit = chi2.ppf(1 - alpha, 2 * n_channels) / (2 * M * n_channels)
return mcsm_values, F, csmNcrit

def calculate_d_amsc(data, tj, fs, alpha=0.05):
if data.ndim == 1 or data.shape[1] < 2: return _calculate_univariate_msc(data, tj, fs, alpha)
n_derivations = data.shape[1]
all_mscs = []
F, crit = None, None
for i in range(n_derivations):
derivation_data = data[:, i]
msc_values, F_temp, crit_temp = _calculate_univariate_msc(derivation_data, tj, fs, alpha)
all_mscs.append(msc_values)
if F is None: F, crit = F_temp, crit_temp
avg_msc = np.mean(np.array(all_mscs), axis=0)
return avg_msc, F, crit

def calculate_d_acsm(data, tj, fs, alpha=0.05):
if data.ndim == 1 or data.shape[1] < 2: return _calculate_univariate_csm(data, tj, fs, alpha)
n_derivations = data.shape[1]
all_csms = []
F, crit = None, None
for i in range(n_derivations):
derivation_data = data[:, i]
csm_values, F_temp, crit_temp = _calculate_univariate_csm(derivation_data, tj, fs, alpha)
all_csms.append(csm_values)
if F is None: F, crit = F_temp, crit_temp
avg_csm = np.mean(np.array(all_csms), axis=0)
return avg_csm, F, crit

def calculate_detection_rates(metric_values, freq_vector, critical_value, signal_freqs, noise_freqs):
detections = {freq: metric_values[np.argmin(np.abs(freq_vector - freq))] > critical_value for freq in signal_freqs + noise_freqs}
tp_count = sum(1 for freq in signal_freqs if detections.get(freq, False))
fp_count = sum(1 for freq in noise_freqs if detections.get(freq, False))
dr = tp_count / len(signal_freqs) if signal_freqs else 0
fpr = fp_count / len(noise_freqs) if noise_freqs else 0
return dr, fpr

def create_derivation_matrix(num_channels):
num_derivations = num_channels * (num_channels + 1) // 2
A = np.zeros((num_derivations, num_channels))
current_row = 0
for i in range(num_channels):
A[current_row, i] = 1
current_row += 1
for i, j in itertools.combinations(range(num_channels), 2):
A[current_row, i] = 1
A[current_row, j] = -1
current_row += 1
return A

def plot_split_coherence_analysis(data_original, data_processed, tj, fs, title, subtitle_info, base_filename, output_dir, signal_freqs, noise_freqs, analysis_mode):
NUM_SPLITS = 5
total_samples = data_original.shape[0]
samples_per_split = total_samples // NUM_SPLITS

code
Code
download
content_copy
expand_less
if samples_per_split < tj:
    tqdm.write(f"\nAVISO: Não há dados suficientes para criar {NUM_SPLITS} segmentos em '{base_filename}'. Pulando.")
    return

if analysis_mode == 'all_derivations':
    metrics_to_plot = {
        "D-aMSC": (calculate_d_amsc, 'firebrick', 'darkblue'),
        "D-aCSM": (calculate_d_acsm, 'purple', 'darkcyan')
    }
else:
    n_signals = data_original.shape[1] if data_original.ndim > 1 else 1
    metric_name = "MMSC" if n_signals > 1 else "MSC"
    phase_metric_name = "MCSM" if n_signals > 1 else "CSM"
    metrics_to_plot = {
        metric_name: (calculate_mmsc, 'firebrick', 'darkblue'),
        phase_metric_name: (calculate_mcsm, 'purple', 'darkcyan')
    }

try:
    for metric_name, (calc_func, color_orig, color_proc) in metrics_to_plot.items():
        fig, axs = plt.subplots(NUM_SPLITS, 2, figsize=(24, 30), sharex=True, sharey=True)
        fig.suptitle(f'{title} - Métrica: {metric_name}', fontsize=22, y=0.99)
        for i in range(NUM_SPLITS):
            start_idx = i * samples_per_split
            end_idx = (i + 1) * samples_per_split
            split_data_orig, split_data_proc = data_original[start_idx:end_idx], data_processed[start_idx:end_idx]
            metric_orig, F, crit_orig = calc_func(split_data_orig, tj, fs)
            metric_proc, _, crit_proc = calc_func(split_data_proc, tj, fs)
            dr_orig, fpr_orig = calculate_detection_rates(metric_orig, F, crit_orig, signal_freqs, noise_freqs)
            dr_proc, fpr_proc = calculate_detection_rates(metric_proc, F, crit_proc, signal_freqs, noise_freqs)
            axs[i, 0].plot(F, metric_orig, color=color_orig, lw=1.5); axs[i, 0].axhline(y=crit_orig, color='k', linestyle='--')
            axs[i, 0].set_title(f'Segmento {i+1} (Original) | DR: {dr_orig:.1%} | FPR: {fpr_orig:.1%}', fontsize=14)
            axs[i, 1].plot(F, metric_proc, color=color_proc, lw=1.5); axs[i, 1].axhline(y=crit_proc, color='k', linestyle='--')
            axs[i, 1].set_title(f'Segmento {i+1} (Processado) | DR: {dr_proc:.1%} | FPR: {fpr_proc:.1%}', fontsize=14)
            axs[i, 0].set_ylabel(f'Coerência\nSegmento {i+1}', fontsize=12)
        for ax in axs.flat:
            ax.grid(True, linestyle='--', alpha=0.6); ax.set_xlim(30, 350); ax.set_ylim(-0.05, 1.05)
            for freq in signal_freqs: ax.axvline(x=freq, color='green', linestyle=':', alpha=0.8)
            for freq in noise_freqs: ax.axvline(x=freq, color='red', linestyle=':', alpha=0.8)
        axs[-1, 0].set_xlabel('Frequência (Hz)', fontsize=14); axs[-1, 1].set_xlabel('Frequência (Hz)', fontsize=14)
        fig.text(0.5, 0.01, subtitle_info, ha='center', fontsize=16, va='bottom', wrap=True)
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(os.path.join(output_dir, f'{base_filename}_{metric_name}.png')); plt.close(fig)
except Exception as e:
    tqdm.write(f"\nERRO ao gerar gráficos para '{base_filename}': {e}")

def parse_filename(filepath):
possible_intensities = ['70dB', '60dB', '50dB', '40dB', '30dB', 'ESP']
filename_stem = os.path.splitext(os.path.basename(filepath))[0]
for intensity in possible_intensities:
if filename_stem.endswith(intensity): return filename_stem[:-len(intensity)], intensity
return None, None

==============================================================================
FUNÇÃO PRINCIPAL
==============================================================================

def main(config):
for key, value in config.items(): globals()[key] = value
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_output_dir = os.path.join(BASE_OUTPUT_DIR, f"run_{timestamp}")
os.makedirs(run_output_dir, exist_ok=True)
save_config_to_yaml(config, run_output_dir)
print(f"--- INICIANDO PROCESSAMENTO (v24.6.3) ---\n  - Saídas em: '{run_output_dir}'")
ELECTRODE_LIST = ['Fz','F3','F4','F7','Fcz','Cz','C3','C4','Pz','P3','P4','T3','T4','T5','T6','Oz']
ELECTRODE_MAP = {name: i for i, name in enumerate(ELECTRODE_LIST)}
all_mat_files = glob.glob(os.path.join(EXPERIMENTAL_DATA_DIR, '*.mat'))

code
Code
download
content_copy
expand_less
for file_path in tqdm(all_mat_files, desc="Processando arquivos"):
    volunteer, intensity = parse_filename(file_path)
    if not volunteer or not intensity: continue
    try:
        base_plot_filename = f'{volunteer}_{intensity}'
        if SKIP_EXISTING_PLOTS and any(glob.glob(os.path.join(run_output_dir, f'{base_plot_filename}_*.png'))): continue
        with h5py.File(file_path, 'r') as f: raw_data, fs = f['x'][:], f['Fs'][0, 0]
        if ANALYSIS_MODE == 'all_derivations':
            source_channels = CHANNELS_TO_USE
            montage_desc = f"Montagem: Todas Derivações de ({', '.join(source_channels)})"
            source_indices = [ELECTRODE_MAP[ch] for ch in source_channels if ch in ELECTRODE_MAP]
            source_data = raw_data[source_indices, :, :]
            A = create_derivation_matrix(len(source_indices))
            derived_data = np.einsum('id,dwt->iwt', A, source_data)
            eeg_windows = derived_data.transpose(1, 2, 0)
        elif ANALYSIS_MODE == 'dipole':
            dipole_names = [f"{ch_pair[0]}-{ch_pair[1]}" for ch_pair in CHANNELS_TO_USE]
            montage_desc = f"Montagem: Dipolo(s) ({', '.join(dipole_names)})"
            analysis_signals = [raw_data[ELECTRODE_MAP[p[0]]] - raw_data[ELECTRODE_MAP[p[1]]] for p in CHANNELS_TO_USE]
            eeg_windows = np.stack(analysis_signals, axis=-1)
        elif ANALYSIS_MODE == 'channel':
            montage_desc = f"Montagem: Canal(is) ({', '.join(CHANNELS_TO_USE)})"
            channel_indices = [ELECTRODE_MAP[ch] for ch in CHANNELS_TO_USE if ch in ELECTRODE_MAP]
            eeg_windows = np.stack([raw_data[idx] for idx in channel_indices], axis=-1)
        else: raise ValueError(f"ANALYSIS_MODE '{ANALYSIS_MODE}' inválido.")
        eeg_processed = eeg_windows.copy()
        if FILTER_TYPE == 'custom':
            b, a = butter(CUSTOM_FILTER_ORDER, [CUSTOM_FILTER_LOW_CUT, CUSTOM_FILTER_HIGH_CUT], 'bandpass', fs=fs)
            eeg_processed = filtfilt(b, a, eeg_processed, axis=1)
        if APPLY_NOTCH_FILTER:
            b_n, a_n = iirnotch(NOTCH_FREQ, NOTCH_Q, fs)
            eeg_processed = filtfilt(b_n, a_n, eeg_processed, axis=1)
        if APPLY_MEAN_REMOVAL and eeg_processed.ndim == 3:
            eeg_processed -= np.mean(eeg_processed, axis=1, keepdims=True)
        num_windows_before = eeg_processed.shape[0]
        if ARTIFACT_REJECTION_TYPE == 'percentile' and num_windows_before > 0:
            axis_to_reduce = (1, 2) if eeg_processed.ndim == 3 else 1
            peak_abs = np.max(np.abs(eeg_processed), axis=axis_to_reduce)
            threshold = np.percentile(peak_abs, 100 - PERCENTILE_REMOVAL_AMOUNT)
            windows_to_keep = peak_abs <= threshold
            eeg_processed, eeg_windows_for_analysis = eeg_processed[windows_to_keep], eeg_windows[windows_to_keep]
        else: eeg_windows_for_analysis = eeg_windows
        if REMOVE_FIRST_TWO_WINDOWS and eeg_processed.shape[0] > 2:
            eeg_processed, eeg_windows_for_analysis = eeg_processed[2:], eeg_windows_for_analysis[2:]
        if eeg_processed.shape[0] < 10: continue
        proc_filt = f"Filtro: {FILTER_TYPE} ({CUSTOM_FILTER_LOW_CUT}-{CUSTOM_FILTER_HIGH_CUT} Hz, O:{CUSTOM_FILTER_ORDER})" if FILTER_TYPE else "Filtro: Não"
        proc_notch = f"Notch 60Hz: {'Sim' if APPLY_NOTCH_FILTER else 'Não'}"
        proc_artifact = f"Rej. Artefato: {ARTIFACT_REJECTION_TYPE} ({PERCENTILE_REMOVAL_AMOUNT}%)" if ARTIFACT_REJECTION_TYPE else "Rej. Artefato: Não"
        subtitle = f"{montage_desc} | {proc_filt} | {proc_notch} | {proc_artifact} | Janelas: {eeg_processed.shape[0]}/{num_windows_before} | Fs: {fs:.0f} Hz"
        n_signals = eeg_processed.shape[-1] if eeg_processed.ndim > 1 else 1
        eeg_processed_cont = eeg_processed.reshape(-1, n_signals) if n_signals > 1 else eeg_processed.flatten()
        eeg_original_cont = eeg_windows_for_analysis.reshape(-1, n_signals) if n_signals > 1 else eeg_windows_for_analysis.flatten()
        if RUN_COHERENCE_PLOTS:
            plot_split_coherence_analysis(eeg_original_cont, eeg_processed_cont, int(fs), fs, f'Análise de Coerência para {volunteer} ({intensity})', subtitle, base_plot_filename, run_output_dir, SIGNAL_FREQUENCIES, NOISE_FREQUENCIES, ANALYSIS_MODE)
    except Exception as e:
        tqdm.write(f"\nERRO CRÍTICO ao processar '{volunteer}_{intensity}': {e}")
print(f"\n\nProcessamento concluído.")
if name == 'main':
parser = argparse.ArgumentParser(description="Executa o pipeline de análise de coerência de EEG.")
parser.add_argument('--config', type=str, help='Caminho para o arquivo de configuração .yml.')
args = parser.parse_args()
final_config = load_config_from_yaml(args.config) if args.config else DEFAULT_CONFIG
main(final_config)

===
CODE B:
import numpy as np
from scipy.stats import chi2, f
from scipy.optimize import minimize
from scipy.io import loadmat, savemat
from scipy.signal import detrend
import matplotlib.pyplot as plt
import os
import warnings
from typing import Dict, List, Tuple, Any
from functools import partial
import multiprocessing
from tqdm import tqdm # Import the progress bar library

Numba is used selectively on compatible functions

from numba import njit, float64, int64, complex128, types

--- 0. Configuration and Global Parameters ---

class Config:
# Production-level run counts for full analysis
N_RUNS_OPTIMIZATION = 1000
N_RUNS_VC_MSC = 10000
ALPHA_DEFAULT = 0.05
FP_DESEJADO_DEFAULT = 0.05
MAX_ITER_FMINCG = 50
TJ_DEFAULT = 32
BIN_DEFAULT = 8
GANHO = 200
REMOC_THRESHOLD = 0.1 / GANHO
DATA_PATH = "C:\Users\alexa\experimental_data\todos\ENTRADAS_PATRICIA"
AVAILABLE_ELECTRODES: List[str] = ['Fz', 'F3', 'F4', 'F7', 'Fcz', 'Cz', 'C3', 'C4', 'Pz', 'P3', 'P4', 'T3', 'T4', 'T5', 'T6', 'Oz']
MAX_WINDOWS_PER_INTENSITY: Dict[str, int] = {'70dB': 50, '60dB': 40, '50dB': 240, '40dB': 440, '30dB': 440, 'ESP': 20}

--- 1. JIT-Compiled Helper Functions (Numba-Safe) ---

@njit(float64:,:, cache=True)
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

@njit(float64[:](complex128[:,:], int64), cache=True)
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

@njit(types.UniTuple(int64, 2)(float64[:], int64[:], float64, float64, float64[:]), cache=True)
def ETS(ord_values: np.ndarray, MM_indices: np.ndarray, alfa: float, NDC: float, vc_msc_critical_values: np.ndarray) -> Tuple[int, int]:
NDC_ceil = int(np.ceil(NDC))
det = ord_values > vc_msc_critical_values
count, dr, time = 0, 0, MM_indices[-1]
for ii in range(len(MM_indices)):
if det[ii]:
count += 1
else:
count = 0
if count >= NDC_ceil:
dr = 1
time = MM_indices[ii]
break
return dr, time

--- 2. FFT-Based Functions (Parallelized with Multiprocessing) ---

def _monte_carlo_single_run(M_int: int, N_window: int, bin_idx: int) -> float:
x = np.random.randn(N_window * M_int)
aux_fft_reshaped = np.fft.fft(x.reshape(N_window, M_int), axis=0)
relevant_fft_slice = aux_fft_reshaped[bin_idx, :].reshape(1, M_int)
aux_msc = msc_fft(relevant_fft_slice, M_int)
return aux_msc[0]

def _calculate_vc_for_single_m(M: int, n_runs: int, n_window: int, bin_idx: int, alfa: float) -> float:
if M <= 1:
return 1.0
ord_values = [_monte_carlo_single_run(M, n_window, bin_idx) for _ in range(n_runs)]
return np.quantile(ord_values, 1.0 - alfa)

def VC_MSC(M_values: np.ndarray, alfa: float, nRuns: int = Config.N_RUNS_VC_MSC) -> Tuple[np.ndarray, np.ndarray]:
N_window, bin_idx = 32, 7
task = partial(_calculate_vc_for_single_m, n_runs=nRuns, n_window=N_window, bin_idx=bin_idx, alfa=alfa)

code
Code
download
content_copy
expand_less
if len(M_values) > 1 and nRuns > 100: # Only parallelize expensive computations
    with multiprocessing.Pool() as pool:
        VC_MC = pool.map(task, M_values.astype(np.int64))
else:
    VC_MC = [task(m) for m in M_values.astype(np.int64)]
    
VC_teorico = np.where(M_values > 1, 1 - alfa**(1.0 / (M_values - 1)), 1.0)
return np.array(VC_MC), VC_teorico

def _generate_ord_sim_single_run(run_index: int, Mmax: int, tj: int, bin_idx: int, Ntotal: int) -> np.ndarray:
ord_sim_row = np.zeros(Mmax)
x = np.random.randn(Ntotal)
fft_x_reshaped = np.fft.fft(x.reshape(tj, Mmax), axis=0)
for M in range(2, Mmax + 1):
ord_sim_row[M-1] = msc_fft(fft_x_reshaped[bin_idx, :M].reshape(1, M), M)[0]
return ord_sim_row

--- 3. Orchestrator Functions (Standard Python) ---

def generate_protocol_parameters(Mmax: int) -> np.ndarray:
P = []
for M_step in range(1, Mmax):
for Mmin in range(2, Mmax):
k = (Mmax - Mmin) / M_step
if k == int(k) and k >= 0:
P.append([Mmin, M_step, Mmax])
P.append([Mmax, 1, Mmax])
return np.array(P, dtype=np.int64)

def funcao_NDC_alfaCorrigido_Mmax(nRuns: int, Mmax: int, alfa_teste: float, FP_desejado: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
tj, bin_idx, Ntotal = Config.TJ_DEFAULT, Config.BIN_DEFAULT, Mmax * Config.TJ_DEFAULT
print("Generating simulated detector outputs (in parallel)...")

code
Code
download
content_copy
expand_less
task = partial(_generate_ord_sim_single_run, Mmax=Mmax, tj=tj, bin_idx=bin_idx, Ntotal=Ntotal)
with multiprocessing.Pool() as pool:
    # Use tqdm to show progress for this pool operation
    results = list(tqdm(pool.imap(task, range(nRuns)), total=nRuns, desc="Simulating Outputs"))
ord_sim = np.array(results)

P = generate_protocol_parameters(Mmax)
alfa_corrigido = np.full(P.shape[0], np.nan)
NDC_minimo = np.full(P.shape[0], np.nan)
num_tests = np.ceil((Mmax - P[:, 0]) / P[:, 1]) + 1
P = P[np.argsort(num_tests), :]

print("Optimizing protocol parameters...")
# --- ETA FEATURE: Wrap the main loop with tqdm ---
for ii in tqdm(range(P.shape[0]), desc="Optimizing Protocols"):
    Mmin, Mstep, current_Mmax = P[ii, :]
    MM_indices = np.arange(Mmin, current_Mmax + 1, Mstep)
    NDC_minimo[ii], _ = estimarNDC(1, alfa_teste, FP_desejado, ord_sim, Mmin, Mstep, current_Mmax)
    
    cost_func = lambda alpha: funcao_custo_v2(alpha, NDC_minimo[ii], MM_indices, ord_sim, FP_desejado)
    res = minimize(cost_func, np.array([alfa_teste]), jac=True, method='CG', options={'maxiter': Config.MAX_ITER_FMINCG})
    alfa_corrigido[ii] = res.x[0]

return alfa_corrigido, NDC_minimo, np.zeros_like(alfa_corrigido), P

def estimarNDC(NDCinicial: int, alfa_teste: float, FP_desejado: float, ord_sim: np.ndarray, Mmin: int, Mstep: int, Mmax: int) -> Tuple[float, np.ndarray]:
MM_indices = np.arange(Mmin, Mmax + 1, Mstep)
NNTmax = len(MM_indices)
if NNTmax == 0: return 1.0, np.array([])
_, vc_teorico = VC_MSC(MM_indices, alfa_teste, nRuns=Config.N_RUNS_VC_MSC)
ord_for_protocol = ord_sim[:, MM_indices - 1]
FP_values = np.zeros(NNTmax + 1)
for NDC_val in range(1, NNTmax + 1):
dr = np.zeros(ord_sim.shape[0])
for ii in range(ord_sim.shape[0]):
dr[ii], _ = ETS(ord_for_protocol[ii, :], MM_indices, alfa_teste, float(NDC_val), vc_teorico)
FP_values[NDC_val] = np.mean(dr)
if FP_values[NDC_val] < FP_desejado:
break
ind = np.where((FP_values < FP_desejado) & (FP_values != 0))[0]
NDC = float(ind[0]) if ind.size > 0 else float(NNTmax)
return NDC, FP_values[1:]

def funcao_custo_v2(alpha, NDC, MM_indices, ord_sim, FP_desejado):
nRuns = ord_sim.shape[0]
alpha_float = np.clip(alpha[0], np.finfo(float).eps, 1 - np.finfo(float).eps)

code
Code
download
content_copy
expand_less
vc = np.where(MM_indices > 1, 1 - alpha_float**(1.0 / (MM_indices - 1)), 1.0)
dr = np.zeros(nRuns)
ord_protocol = ord_sim[:, MM_indices - 1]
for ii in range(nRuns):
    dr[ii], _ = ETS(ord_protocol[ii, :], MM_indices, alpha_float, NDC, vc)
    
FP = np.mean(dr)
erro = FP - FP_desejado
return 0.5 * erro**2, np.array([erro])

def protocolo_deteccao(x: np.ndarray, parametros: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
num_fft_points, num_windows = x.shape
max_freq_bins = num_fft_points // 2 + 1
ord_values = np.zeros((max_freq_bins, num_windows + 1))
xfft = np.fft.fft(x, axis=0)
for M in range(2, num_windows + 1):
ord_values[:, M] = msc_fft(xfft[:max_freq_bins, :M], M)

code
Code
download
content_copy
expand_less
dr = np.zeros((max_freq_bins, parametros.shape[0]))
time = np.full_like(dr, -1, dtype=np.int64)

for ii in range(parametros.shape[0]):
    Mmin, Mstep, Mmax_p, NDC_p, alfa_p = parametros[ii, :]
    MM_indices = np.arange(int(Mmin), int(Mmax_p) + 1, int(Mstep))
    if MM_indices.size == 0: continue
    vc = np.where(MM_indices > 1, 1 - alfa_p**(1.0 / (MM_indices - 1)), 1.0)
    for ll in range(max_freq_bins):
        dr[ll, ii], time[ll, ii] = ETS(ord_values[ll, MM_indices], MM_indices, alfa_p, NDC_p, vc)
        
return dr, time
--- 4. Main Execution Block ---

def main_analysis():
print("--- Starting Hybrid-Optimized EEG Analysis ---")
target_electrodes = ['F3', 'F4', 'Fz', 'T5', 'T6', 'Pz', 'Cz', 'Oz']
dipole_labels = generate_dipole_labels(target_electrodes)
num_dipoles = len(dipole_labels)
print(f"Analyzing {num_dipoles} derivations from {len(target_electrodes)} electrodes.")

code
Code
download
content_copy
expand_less
Vvoluntario = ['Ab', 'An', 'Bb', 'Er', 'Lu', 'So', 'Qu', 'Vi', 'Sa', 'Ti', 'Wr']
Intensidade = '50dB'
Mmax_global = Config.MAX_WINDOWS_PER_INTENSITY[Intensidade]

alfa_opt, fp_desejado_opt = Config.ALPHA_DEFAULT, Config.FP_DESEJADO_DEFAULT
filename = f"Hybrid_Optimized_Params_Mmax{Mmax_global}.mat"
filepath = os.path.join(Config.DATA_PATH, filename)

if not os.path.exists(filepath):
    print(f"Pre-computed file '{filename}' not found. Generating now (this may take a while)...")
    alfa_c, ndc_m, _, P_calc = funcao_NDC_alfaCorrigido_Mmax(Config.N_RUNS_OPTIMIZATION, Mmax_global, alfa_opt, fp_desejado_opt)
    savemat(filepath, {'alfa_corrigido': alfa_c, 'NDC_minimo': ndc_m, 'P': P_calc})
    loaded_data = {'alfa_corrigido': alfa_c, 'NDC_minimo': ndc_m, 'P': P_calc}
else:
    print(f"Loading pre-computed parameters from: {filepath}")
    loaded_data = loadmat(filepath)

parametros = np.hstack((loaded_data['P'], loaded_data['NDC_minimo'][:, np.newaxis], loaded_data['alfa_corrigido'][:, np.newaxis]))

Tdr_all, Ttime_all = [], []
electrode_indices = [Config.AVAILABLE_ELECTRODES.index(e) for e in target_electrodes]

for voluntario in tqdm(Vvoluntario, desc="Processing Volunteers"):
    data_file = os.path.join(Config.DATA_PATH, f"{voluntario}{Intensidade}.mat")
    if not os.path.exists(data_file): continue
    
    vol_data = loadmat(data_file)
    x_raw, binsM = vol_data['x'], vol_data['binsM'].flatten()
    x_subset = x_raw[:, :, electrode_indices]
    
    time_pts, n_windows, n_chans = x_subset.shape
    bipolar_data = np.zeros((time_pts, n_windows, num_dipoles))
    for i in range(n_windows):
        bipolar_data[:, i, :] = dipolos(x_subset[:, i, :])

    dr_volunteer, time_volunteer = [], []
    for ndipole in range(num_dipoles):
        x_channel = bipolar_data[:, :, ndipole]
        x_channel -= np.mean(x_channel, axis=0, keepdims=True)
        noisy = np.max(np.abs(x_channel), axis=0) > Config.REMOC_THRESHOLD
        x_clean = x_channel[:, ~noisy][:, :Mmax_global]
        if x_clean.shape[1] < 2: continue
        
        dr, time = protocolo_deteccao(x_clean, parametros)
        dr_volunteer.append(dr)
        time_volunteer.append(time)
        
    Tdr_all.append(np.stack(dr_volunteer, axis=-1))
    Ttime_all.append(np.stack(time_volunteer, axis=-1))

Tdr_overall = np.stack(Tdr_all, axis=-1)

binsM_0_idx = binsM - 1
binsR_0_idx = np.setdiff1d(np.arange(Tdr_overall.shape[0]), binsM_0_idx)

TXD_per_protocol = np.mean(Tdr_overall[binsM_0_idx, ...], axis=(0, 2, 3)) * 100
FP_per_protocol = np.mean(Tdr_overall[binsR_0_idx, ...], axis=(0, 2, 3)) * 100

plt.figure(figsize=(12, 7))
plt.plot(TXD_per_protocol, 'o-', label='Detection Rate (PD)')
plt.plot(FP_per_protocol, 'o-', label='False Positive Rate (FP)')
plt.title(f'Detection Performance vs. Protocol ({Intensidade})')
plt.xlabel('Protocol Parameter Set Index')
plt.ylabel('Rate (%)')
plt.grid(True, which='both', linestyle='--')
plt.legend()
plt.show()

if name == 'main':
multiprocessing.freeze_support()

code
Code
download
content_copy
expand_less
if not os.path.exists(Config.DATA_PATH):
    print(f"ERROR: DATA_PATH does not exist: {Config.DATA_PATH}")
else:
    main_analysis()

===

===
CODE C:

=============================================================================
Section 1: Imports and Configuration
=============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt

You must install this library first by running: pip install mat73

import mat73

--- User-editable paths ---
Path to the directory containing the 'NDC_AlfaCorrigido_*.mat' file

path_mat_file = r"C:\PPGEE\Assessing CGST on ASSR\Numero_Deteccoes_consecutiva_H"

Path to the directory containing the EEG data (.mat files for each volunteer)

path_eeg_data = r"C:\Users\alexa\experimental_data\todos\ENTRADAS_PATRICIA"

=============================================================================
Section 2: Function Definitions (Replication of .m files)
=============================================================================

def VC_MSC_py(M, alfa):
"""
Calculates the theoretical critical value for the Magnitude-Squared Coherence (MSC).
Equivalent to VC_MSC.m
"""
M_safe = np.asarray(M, dtype=float)
invalid_mask = M_safe <= 1
M_safe[invalid_mask] = np.nan
vc_teorico = 1 - np.power(alfa, 1. / (M_safe - 1))
return vc_teorico

def msc_fft_py(Y):
"""
Calculates the Magnitude-Squared Coherence (MSC) from FFT data.
Equivalent to msc_fft.m
"""
M = Y.shape[1]
if M == 0:
return np.full((Y.shape[0],), np.nan)
numerator = np.abs(np.sum(Y, axis=1))**2
denominator = M * np.sum(np.abs(Y)**2, axis=1)
ord_val = np.zeros_like(numerator, dtype=float)
non_zero_den_mask = denominator != 0
ord_val[non_zero_den_mask] = numerator[non_zero_den_mask] / denominator[non_zero_den_mask]
return ord_val

def ETS_py(ord_values, MM, alfa, NDC):
"""
Implements the Early Termination Strategy for sequential tests.
Equivalent to ETS.m
"""
NDC = np.ceil(NDC)
valor_critico = VC_MSC_py(MM, alfa)
det = ord_values[MM - 2] > valor_critico
consecutive_detections = 0
detection_made = 0
detection_time = MM[-1]
for i, is_detected in enumerate(det):
if is_detected:
consecutive_detections += 1
else:
consecutive_detections = 0
if consecutive_detections >= NDC:
detection_made = 1
detection_time = MM[i]
break
return detection_made, detection_time

def protocolo_deteccao_py(x, parametros):
"""
Applies the detection protocol to input data 'x'.
Equivalent to protocolo_deteccao.m
"""
binsM_protocol = 120
num_windows_total = x.shape[1]
xfft = np.fft.fft(x, axis=0)
ord_matrix = np.zeros((binsM_protocol, num_windows_total - 1))
for M in range(2, num_windows_total + 1):
xfft_slice = xfft[:binsM_protocol, :M]
ord_matrix[:, M - 2] = msc_fft_py(xfft_slice)
num_param_sets = parametros.shape[0]
dr_results = np.zeros((binsM_protocol, num_param_sets))
time_results = np.zeros((binsM_protocol, num_param_sets))
for ii in range(num_param_sets):
Mmin, Mstep, Mmax, NDC, alfa_corr = parametros[ii, :]
MM = np.arange(Mmin, Mmax + 1, Mstep, dtype=int)
if MM.size == 0: continue
for ll in range(binsM_protocol):
dr, time = ETS_py(ord_matrix[ll, :], MM, alfa_corr, NDC)
dr_results[ll, ii] = dr
time_results[ll, ii] = time
return dr_results, time_results

def pareto_front_py(points):
"""
Finds the Pareto front for a set of points.
Assumes maximization of both objectives.
"""
n_points = points.shape[0]
is_pareto = np.ones(n_points, dtype=bool)
for i in range(n_points):
for j in range(n_points):
if i == j: continue
if np.all(points[j] >= points[i]) and np.any(points[j] > points[i]):
is_pareto[i] = False
break
return points[is_pareto], np.where(is_pareto)[0]

=============================================================================
Section 3: Main Script (Replication of sinal_eeg.m)
=============================================================================
--- Script Parameters ---

Vvoluntario = ['Ab', 'An', 'Bb', 'Er', 'Lu', 'So', 'Qu', 'Vi', 'Sa', 'Ti', 'Wr']
Intensidade = ['50dB']
Mmax = 240
alfa = 0.05
FP_desejado = 0.05
pos_ele = 1
ganho = 200
remoc = 0.1 / ganho

--- Load Protocol Parameters ---

mat_filename = f"NDC_AlfaCorrigido_Mmax{Mmax}alfa{alfa}_FPdesejado{FP_desejado}.mat"
mat_filepath = os.path.join(path_mat_file, mat_filename)

try:
print(f"Loading parameters from: {mat_filepath}")
mat_contents = mat73.loadmat(mat_filepath)
P = mat_contents['P']
alfa_corrigido = mat_contents['alfa_corrigido']
NDC_minimo = mat_contents['NDC_minimo']
except FileNotFoundError:
print(f"FATAL ERROR: Could not find the parameter file:\n{mat_filepath}")
exit()
except KeyError as e:
print(f"FATAL ERROR: The .mat file is missing an expected variable: {e}")
exit()

--- Assemble Parameters Matrix ---

parametros = np.hstack((P, np.array(NDC_minimo).reshape(-1, 1), np.array(alfa_corrigido).reshape(-1, 1)))

--- Initialize Result Storage Arrays ---

num_bins_protocol = 120
num_params = parametros.shape[0]
num_volunteers = len(Vvoluntario)
Tdr = np.zeros((num_bins_protocol, num_params, num_volunteers))
Ttime = np.zeros((num_bins_protocol, num_params, num_volunteers))

--- Main Loop over Volunteers ---

for cont_vol, voluntario_code in enumerate(Vvoluntario):
intensidade = Intensidade[0]
eeg_filename = f"{voluntario_code}{intensidade}.mat"
eeg_filepath = os.path.join(path_eeg_data, eeg_filename)

code
Code
download
content_copy
expand_less
print(f"Processing volunteer {cont_vol + 1}/{num_volunteers}: {eeg_filename}...")

eeg_data = mat73.loadmat(eeg_filepath)
x = eeg_data['x']
Fs = float(eeg_data['Fs'])

# MODIFIED: Convert the indexing array to an integer type to prevent IndexError.
binsM_analysis = (np.array(eeg_data['binsM']).flatten() - 1).astype(int)

# --- Preprocessing ---
x = x[:, :, pos_ele - 1]
x = x - x.mean(axis=0, keepdims=True)
x = x[:, 2:]
Vmax = np.max(np.abs(x), axis=0)
clean_indices = Vmax <= remoc
x = x[:, clean_indices]
x = x[:, :Mmax]
 
if x.shape[1] < 2:
    print(f"  -> Skipping volunteer {voluntario_code} due to insufficient clean data (<2 windows).")
    continue
    
dr, time = protocolo_deteccao_py(x, parametros)
Tdr[:, :, cont_vol] = dr
Ttime[:, :, cont_vol] = time

print("\nProcessing complete. Starting analysis and plotting...")

=============================================================================
Section 4: Analysis and Plotting
=============================================================================
--- Calculate TXD (Detection Rate) and FP (False Positive) ---
This line will now work correctly with the integer-type binsM_analysis

TXD = np.mean(np.mean(Tdr[binsM_analysis, :, :], axis=2), axis=0)
all_bins = np.arange(100)
binsR = np.setdiff1d(all_bins, binsM_analysis)
binsR = binsR[binsR > 1]
FP = np.mean(np.mean(Tdr[binsR, :, :], axis=2), axis=0)

--- Plot 1: Detection Rate ---

plt.figure(figsize=(8, 6))
plt.plot(TXD, '.k', markersize=10, label='Parameter Sets')
plt.axhline(y=TXD[-1], color='r', linestyle=':', linewidth=2, label=f'Final Set TXD: {TXD[-1]:.2f}')
plt.ylabel('Taxa de Detecção (Detection Rate)')
plt.xlabel('Parameter Set Index')
plt.title('Detection Rate per Parameter Set')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

--- Plot 2: False Positive Rate ---

plt.figure(figsize=(8, 6))
plt.plot(FP, '.k', markersize=10, label='Parameter Sets')
plt.axhline(y=FP_desejado, color='r', linestyle=':', linewidth=2, label=f'Desired FP: {FP_desejado:.2f}')
plt.ylabel('Falso Positivo (False Positive Rate)')
plt.xlabel('Parameter Set Index')
plt.title('False Positive Rate per Parameter Set')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

--- Plot 3: Pareto Front (Detection Rate vs. Time) ---

timeM = Ttime[binsM_analysis, :, :]
timeM[timeM == -1] = Mmax
timeM_mean = np.mean(np.mean(timeM, axis=2), axis=0)
TXD_percent = TXD * 100

fig1, axes1 = plt.subplots(figsize=(12, 8))
axes1.plot(TXD_percent, timeM_mean, '.k', markersize=8, label='All Parameter Sets')

points_for_pareto = np.vstack((TXD_percent, -timeM_mean)).T
p_front_points, p_front_indices = pareto_front_py(points_for_pareto)

sort_order = np.argsort(p_front_points[:, 0])
p_front_points = p_front_points[sort_order]
p_front_indices = p_front_indices[sort_order]

axes1.plot(p_front_points[:, 0], -p_front_points[:, 1], '-or',
markersize=8, linewidth=1.5, label='Pareto Front')

axes1.set_xlabel('Detection Rate (%)', fontsize=12)
axes1.set_ylabel('Mean Exam Time (seconds)', fontsize=12)
axes1.set_title('Performance: Detection Rate vs. Exam Time', fontsize=14)
axes1.grid(True, linestyle='--', alpha=0.6)
axes1.legend()

print('\n--- Pareto Front Optimal Parameters ---')
for idx in p_front_indices:
pd_val = TXD_percent[idx]
time_val = timeM_mean[idx]
matches = np.where((np.isclose(TXD_percent, pd_val)) & (np.isclose(timeM_mean, time_val)))[0]
param_str = ' | '.join([f"Buffer:{int(parametros[m,0])}, M_step:{int(parametros[m,1])}" for m in matches])
print(f'PD = {pd_val:.2f}% | Time = {time_val:.2f}s | Configurations: {param_str}')
text_str = f"{{{int(parametros[matches[0], 0])},{int(parametros[matches[0], 1])}}}"
axes1.text(pd_val, time_val * 0.98, text_str, fontsize=9, ha='center')

if len(p_front_points) > 0:
axes1.set_xlim([np.min(p_front_points[:, 0]) * 0.95, np.max(p_front_points[:, 0]) * 1.05])
axes1.set_ylim([np.min(-p_front_points[:, 1]) * 0.95, Mmax * 1.05])

plt.show()

