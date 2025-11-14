import numpy as np
from scipy.stats import chi2, f as finv
from scipy.optimize import minimize
from scipy.io import loadmat, savemat
from scipy.signal import detrend
import matplotlib.pyplot as plt
import os
import warnings
from typing import Dict, List, Tuple, Any
from functools import partial
import multiprocessing
from tqdm import tqdm
import itertools

# Numba is used selectively on compatible functions
from numba import njit, float64, int64, complex128, types

# --- 0. Configuration and Global Parameters ---
class Config:
    N_RUNS_OPTIMIZATION = 1000 
    N_RUNS_VC_MSC = 10000
    ALPHA_DEFAULT = 0.05
    FP_DESEJADO_DEFAULT = 0.05
    MAX_ITER_FMINCG = 50
    TJ_DEFAULT = 32
    BIN_DEFAULT = 8
    GANHO = 200
    REMOC_THRESHOLD = 0.1 / GANHO
    DATA_PATH = "C:\\Users\\alexa\\experimental_data\\todos\\ENTRADAS_PATRICIA"
    AVAILABLE_ELECTRODES: List[str] = ['Fz', 'F3', 'F4', 'F7', 'Fcz', 'Cz', 'C3', 'C4', 'Pz', 'P3', 'P4', 'T3', 'T4', 'T5', 'T6', 'Oz']
    MAX_WINDOWS_PER_INTENSITY: Dict[str, int] = {'70dB': 50, '60dB': 40, '50dB': 240, '40dB': 440, '30dB': 440, 'ESP': 20}

# =============================================================================
# Section 1: Ported Metrics from Code A
# =============================================================================

def msweep(matrix, r):
    A = np.copy(matrix).astype(np.complex128)
    for k in range(r):
        d = A[k, k]
        if np.abs(d) < 1e-12: d = 1e-12
        col_k, row_k = A[:, k].copy(), A[k, :].copy()
        A -= np.outer(col_k, row_k) / d; A[k, :] = row_k / d; A[:, k] = -col_k / d; A[k, k] = 1.0 / d
    return A

def _calculate_metric_for_window(data_cont, tj, fs, metric_func, alpha=0.05):
    """Helper to call metric functions with continuous data."""
    if data_cont.shape[0] < tj:
        return np.zeros(tj // 2 + 1), 1.0
    metric_values, _, crit = metric_func(data_cont, tj, fs, alpha)
    return metric_values, crit

def _calculate_univariate_msc(data, tj, fs, alpha):
    M = data.shape[0] // tj
    if M <= 1: return np.zeros(tj // 2 + 1), np.arange(tj // 2 + 1) * fs / tj, 1.0
    epochs = data.flatten()[:M*tj].reshape(M, tj)
    nf = tj // 2 + 1
    Y = np.fft.fft(epochs, axis=1)[:, :nf]
    sum_Y = np.sum(Y, axis=0)
    sum_Y_sq_mag = np.sum(np.abs(Y)**2, axis=0)
    msc_values = np.zeros(nf)
    valid_idx = sum_Y_sq_mag > 1e-12
    msc_values[valid_idx] = np.abs(sum_Y[valid_idx])**2 / (M * sum_Y_sq_mag[valid_idx])
    crit = 1 - alpha**(1/(M-1)) if M > 1 else 1.0
    F = np.arange(nf) * fs / tj
    return msc_values, F, crit

def _calculate_univariate_csm(data, tj, fs, alpha):
    M = data.shape[0] // tj
    if M <= 1: return np.zeros(tj // 2 + 1), np.arange(tj // 2 + 1) * fs / tj, 1.0
    epochs = data.flatten()[:M * tj].reshape(M, tj)
    nf = tj // 2 + 1
    Y = np.fft.fft(epochs, axis=1)[:, :nf]
    teta = np.angle(Y)
    sum_cos = np.sum(np.cos(teta), axis=0); sum_sin = np.sum(np.sin(teta), axis=0)
    csm_values = (sum_cos**2 + sum_sin**2) / (M**2)
    crit = chi2.ppf(1 - alpha, 2) / (2 * M) if M > 0 else 1.0
    F = np.arange(nf) * fs / tj
    return csm_values, F, crit

def calculate_mmsc(data, tj, fs, alpha=0.05):
    n_samples, n_channels = data.shape if data.ndim > 1 else (data.shape[0], 1)
    if n_channels == 1: return _calculate_univariate_msc(data, tj, fs, alpha)
    M = n_samples // tj; nf = tj // 2 + 1; F = np.arange(nf) * fs / tj
    if M <= n_channels: return np.zeros(nf), F, 1.0
    epochs = data[:M*tj, :].T.reshape(n_channels, M, tj)
    Sfft = np.fft.fft(epochs, axis=2)[:, :, :nf]; mmsc_values = np.zeros(nf)
    for kf in range(nf):
        Sfft_slice = Sfft[:, :, kf]
        spec_matrix_a = np.zeros((n_channels + 1, n_channels + 1), dtype=np.complex128)
        spec_matrix_a[:n_channels, :n_channels] = Sfft_slice @ Sfft_slice.conj().T
        V = np.sum(Sfft_slice, axis=1)
        spec_matrix_a[n_channels, :n_channels] = V.conj()
        spec_matrix_a[:n_channels, n_channels] = V
        spec_matrix_a[n_channels, n_channels] = 1
        spec_matrix_as = msweep(spec_matrix_a, n_channels)
        mmsc_values[kf] = (1 - np.real(spec_matrix_as[n_channels, n_channels])) / M
    Fcrit = finv.ppf(1-alpha, 2*n_channels, 2*(M-n_channels))
    k2Ncrit = Fcrit / (((M - n_channels) / n_channels) + Fcrit)
    return mmsc_values, F, k2Ncrit

def calculate_mcsm(data, tj, fs, alpha=0.05):
    n_samples, n_channels = data.shape if data.ndim > 1 else (data.shape[0], 1)
    if n_channels == 1: return _calculate_univariate_csm(data, tj, fs, alpha)
    M = n_samples // tj; nf = tj // 2 + 1; F = np.arange(nf) * fs / tj
    if M == 0: return np.zeros(nf), F, 1.0
    epochs = data[:M * tj, :].reshape(M, tj, n_channels)
    Y = np.fft.fft(epochs, axis=1)[:, :nf, :]
    teta = np.angle(Y)
    C_mean_over_channels = np.mean(np.cos(teta), axis=2)
    S_mean_over_channels = np.mean(np.sin(teta), axis=2)
    teta_med = np.arctan2(S_mean_over_channels, C_mean_over_channels)
    sum_cos = np.sum(np.cos(teta_med), axis=0); sum_sin = np.sum(np.sin(teta_med), axis=0)
    mcsm_values = (sum_cos**2 + sum_sin**2) / (M**2)
    csmNcrit = chi2.ppf(1 - alpha, 2 * n_channels) / (2 * M * n_channels)
    return mcsm_values, F, csmNcrit

def calculate_d_amsc(data, tj, fs, alpha=0.05):
    if data.ndim == 1 or data.shape[1] < 2: return _calculate_univariate_msc(data, tj, fs, alpha)
    n_derivations = data.shape[1]; all_mscs = []
    for i in range(n_derivations):
        msc_values, F, crit = _calculate_univariate_msc(data[:, i], tj, fs, alpha)
        all_mscs.append(msc_values)
    return np.mean(np.array(all_mscs), axis=0), F, crit

def calculate_d_acsm(data, tj, fs, alpha=0.05):
    if data.ndim == 1 or data.shape[1] < 2: return _calculate_univariate_csm(data, tj, fs, alpha)
    n_derivations = data.shape[1]; all_csms = []
    for i in range(n_derivations):
        csm_values, F, crit = _calculate_univariate_csm(data[:, i], tj, fs, alpha)
        all_csms.append(csm_values)
    return np.mean(np.array(all_csms), axis=0), F, crit

# --- 2. JIT-Compiled Helper Functions (Numba-Safe) ---
@njit(float64[:,:](float64[:,:]), cache=True)
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
        if denominator[i] > 1e-12:
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

# --- 3. Orchestrator and Protocol Functions ---
def generate_protocol_parameters(Mmax: int) -> np.ndarray:
    P = []
    for M_step in range(1, Mmax):
        for Mmin in range(2, Mmax):
            k = (Mmax - Mmin) / M_step
            if k == int(k) and k >= 0:
                P.append([Mmin, M_step, Mmax])
    P.append([Mmax, 1, Mmax])
    return np.array(P, dtype=np.int64)

def pareto_front(points):
    n_points = points.shape[0]
    is_pareto = np.ones(n_points, dtype=bool)
    for i in range(n_points):
        for j in range(n_points):
            if i == j: continue
            if np.all(points[j] >= points[i]) and np.any(points[j] > points[i]):
                is_pareto[i] = False
                break
    return points[is_pareto], np.where(is_pareto)[0]

def protocolo_deteccao(x: np.ndarray, fs: float, parametros: np.ndarray, metric_name: str) -> Tuple[np.ndarray, np.ndarray]:
    time_pts, num_windows = x.shape[0], x.shape[1]
    if x.ndim > 2: # For dipole data
        time_pts, num_windows, n_chans = x.shape
    
    max_freq_bins = time_pts // 2 + 1
    ord_values = np.zeros((max_freq_bins, num_windows + 1))

    METRIC_FUNCTIONS = {
        'MMSC': calculate_mmsc, 'MCSM': calculate_mcsm,
        'D-aMSC': calculate_d_amsc, 'D-aCSM': calculate_d_acsm
    }
    metric_func = METRIC_FUNCTIONS[metric_name]

    for M in range(2, num_windows + 1):
        # Prepare continuous data for metric functions
        if x.ndim == 3: # Dipole (derivations)
             data_slice_cont = x[:, :M, :].transpose(1,0,2).reshape(-1, x.shape[2])
        else: # Channel
             data_slice_cont = x[:, :M].T.reshape(-1, 1) if x.ndim == 2 else x[:, :M].T.reshape(-1, x.shape[2])
        
        ord_values[:, M], _ = _calculate_metric_for_window(data_slice_cont, time_pts, fs, metric_func)

    dr = np.zeros((max_freq_bins, parametros.shape[0]))
    time = np.full_like(dr, -1, dtype=np.int64)
    
    for ii in range(parametros.shape[0]):
        Mmin, Mstep, Mmax_p, NDC_p, alfa_p = parametros[ii, :]
        MM_indices = np.arange(int(Mmin), int(Mmax_p) + 1, int(Mstep))
        if MM_indices.size == 0: continue
        
        # NOTE: Using theoretical MSC critical value as a proxy.
        # A full implementation would require deriving or simulating critical values for each metric.
        vc = np.where(MM_indices > 1, 1 - alfa_p**(1.0 / (MM_indices - 1)), 1.0)
        
        for ll in range(max_freq_bins):
            dr[ll, ii], time[ll, ii] = ETS(ord_values[ll, MM_indices], MM_indices, alfa_p, NDC_p, vc)
            
    return dr, time

# --- 4. Main Execution Block ---
def main_analysis():
    print("--- Starting Hybrid-Optimized EEG Analysis ---")
    
    # Analysis Parameters
    target_electrodes = ['Fz', 'Cz', 'Pz', 'Oz', 'T3', 'T4']
    metrics_to_run = ['MMSC', 'MCSM', 'D-aMSC', 'D-aCSM']
    Vvoluntario = ['Ab', 'An', 'Er', 'Qu', 'Sa', 'Ti', 'Wr']
    Intensidade = '50dB'
    Mmax_global = Config.MAX_WINDOWS_PER_INTENSITY[Intensidade]

    # Load or compute protocol parameters
    alfa_opt, fp_desejado_opt = Config.ALPHA_DEFAULT, Config.FP_DESEJADO_DEFAULT
    filename = f"Hybrid_Optimized_Params_Mmax{Mmax_global}.mat" # Using MSC-optimized params for all
    filepath = os.path.join(Config.DATA_PATH, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Parameter file not found, please generate it first: {filepath}")
    loaded_data = loadmat(filepath)
    parametros = np.hstack((loaded_data['P'], loaded_data['NDC_minimo'][:, np.newaxis], loaded_data['alfa_corrigido'][:, np.newaxis]))
    
    electrode_indices = [Config.AVAILABLE_ELECTRODES.index(e) for e in target_electrodes]
    
    results = {}

    for metric in metrics_to_run:
        print(f"\n===== Running analysis for metric: {metric} =====")
        Tdr_all, Ttime_all = [], []
        
        for voluntario in tqdm(Vvoluntario, desc=f"Volunteers ({metric})"):
            data_file = os.path.join(Config.DATA_PATH, f"{voluntario}{Intensidade}.mat")
            if not os.path.exists(data_file): continue
            
            vol_data = loadmat(data_file)
            x_raw, binsM, fs = vol_data['x'], vol_data['binsM'].flatten(), float(vol_data['Fs'][0,0])
            x_subset = x_raw[:, :, electrode_indices]
            
            # Select correct data based on metric type
            if metric in ['D-aMSC', 'D-aCSM']:
                time_pts, n_windows, n_chans = x_subset.shape
                num_dipoles = n_chans * (n_chans + 1) // 2
                input_data = np.zeros((time_pts, n_windows, num_dipoles))
                for i in range(n_windows):
                    input_data[:, i, :] = dipolos(x_subset[:, i, :])
            else: # MMSC, MCSM
                input_data = x_subset
            
            # Preprocessing
            input_data -= np.mean(input_data, axis=0, keepdims=True)
            noisy = np.max(np.abs(input_data), axis=0) > Config.REMOC_THRESHOLD
            clean_data = input_data[:, ~noisy.any(axis=1) if input_data.ndim==3 else ~noisy][:, :Mmax_global]
            if clean_data.shape[1] < 2: continue
            
            dr, time = protocolo_deteccao(clean_data, fs, parametros, metric)
            Tdr_all.append(dr)
            Ttime_all.append(time)

        if not Tdr_all:
            print(f"No valid data found for metric {metric}. Skipping.")
            continue

        Tdr_overall = np.stack(Tdr_all, axis=-1)
        Ttime_overall = np.stack(Ttime_all, axis=-1)
        
        # Performance calculation
        binsM_0_idx = binsM - 1
        all_bins = np.arange(Tdr_overall.shape[0])
        binsR_0_idx = np.setdiff1d(all_bins, binsM_0_idx)
        binsR_0_idx = binsR_0_idx[binsR_0_idx > 1]
        
        TXD_per_protocol = np.mean(Tdr_overall[binsM_0_idx, ...], axis=(0, 2)) * 100
        FP_per_protocol = np.mean(Tdr_overall[binsR_0_idx, ...], axis=(0, 2)) * 100
        
        timeM = Ttime_overall[binsM_0_idx, :, :]
        timeM[timeM == -1] = Mmax_global
        timeM_mean = np.mean(timeM, axis=(0, 2))

        results[metric] = {'txd': TXD_per_protocol, 'fpr': FP_per_protocol, 'time': timeM_mean}

    # --- Plotting ---
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    # Plot 1: TPR vs. Time (Pareto)
    fig1, ax1 = plt.subplots(figsize=(14, 9))
    for i, (metric, data) in enumerate(results.items()):
        points = np.vstack((data['txd'], -data['time'])).T
        p_front, _ = pareto_front(points)
        sort_order = np.argsort(p_front[:, 0])
        p_front = p_front[sort_order]
        ax1.plot(p_front[:, 0], -p_front[:, 1], 'o-', color=colors[i], label=f'{metric} Pareto Front')
    
    ax1.set_title(f'Pareto Front Comparison: Detection Rate vs. Exam Time ({Intensidade})', fontsize=16)
    ax1.set_xlabel('True Positive Rate (TPR / Detection Rate, %)', fontsize=12)
    ax1.set_ylabel('Mean Exam Time (s)', fontsize=12)
    ax1.grid(True, which='both', linestyle='--')
    ax1.legend()
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(left=0)
    plt.tight_layout()
    plt.show()

    # Plot 2: FPR vs. Time
    fig2, ax2 = plt.subplots(figsize=(14, 9))
    for i, (metric, data) in enumerate(results.items()):
        sorted_indices = np.argsort(data['time'])
        ax2.plot(data['time'][sorted_indices], data['fpr'][sorted_indices], 'o-', color=colors[i], label=metric, alpha=0.8)

    ax2.set_title(f'Performance Comparison: False Positive Rate vs. Exam Time ({Intensidade})', fontsize=16)
    ax2.set_xlabel('Mean Exam Time (s)', fontsize=12)
    ax2.set_ylabel('False Positive Rate (FPR, %)', fontsize=12)
    ax2.grid(True, which='both', linestyle='--')
    ax2.legend()
    ax2.set_ylim(bottom=0)
    ax2.set_xlim(left=0)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    if not os.path.exists(Config.DATA_PATH):
        print(f"ERROR: DATA_PATH does not exist: {Config.DATA_PATH}")
    else:
        main_analysis()