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

# Numba is used selectively on compatible functions
from numba import njit, float64, int64, complex128, types

# --- 0. Configuration and Global Parameters ---
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
    DATA_PATH = "C:\\Users\\alexa\\experimental_data\\todos\\ENTRADAS_PATRICIA"
    AVAILABLE_ELECTRODES: List[str] = ['Fz', 'F3', 'F4', 'F7', 'Fcz', 'Cz', 'C3', 'C4', 'Pz', 'P3', 'P4', 'T3', 'T4', 'T5', 'T6', 'Oz']
    MAX_WINDOWS_PER_INTENSITY: Dict[str, int] = {'70dB': 50, '60dB': 40, '50dB': 240, '40dB': 440, '30dB': 440, 'ESP': 20}

# --- 1. JIT-Compiled Helper Functions (Numba-Safe) ---

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

# --- 2. FFT-Based Functions (Parallelized with Multiprocessing) ---

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

# --- 3. Orchestrator Functions (Standard Python) ---
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

# --- 4. Main Execution Block ---
def main_analysis():
    print("--- Starting Hybrid-Optimized EEG Analysis ---")
    target_electrodes = ['F3', 'F4', 'Fz', 'T5', 'T6', 'Pz', 'Cz', 'Oz']
    dipole_labels = generate_dipole_labels(target_electrodes)
    num_dipoles = len(dipole_labels)
    print(f"Analyzing {num_dipoles} derivations from {len(target_electrodes)} electrodes.")
    
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

if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    if not os.path.exists(Config.DATA_PATH):
        print(f"ERROR: DATA_PATH does not exist: {Config.DATA_PATH}")
    else:
        main_analysis()