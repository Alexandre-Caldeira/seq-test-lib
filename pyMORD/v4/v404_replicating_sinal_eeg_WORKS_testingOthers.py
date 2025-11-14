# =============================================================================
#
# EEG SEQUENTIAL PROTOCOL ANALYSIS PIPELINE (v4.2.1-fixed)
#
# Author: Caldeira, AG @ NIAS | PPGEE UFMG
# Date: Nov 11 2025
#
# Description:
# This script performs a comparative analysis of different Multivariate Outlier
# Detection (MORD) techniques on EEG data.
#
# THIS VERSION (v4.2.1-fixed) REINTEGRATES THE FULL SUITE OF METRICS
# including MMSC, MCSM, D-aMSC, and D-aCSM, fixing the key error that
# prevented them from running. It also retains the special 'MSC-REPLICA'
# mode to fully replicate the analysis from a reference notebook.
#
# Features:
# - Configurable Analysis: All parameters are managed via a `config.yml` file.
# - Multiple Metrics: Compares MMSC, MCSM, D-aMSC, D-aCSM, and a special MSC-REPLICA.
# - Advanced Preprocessing: Configurable pipeline for filtering and artifact rejection.
# - Performance Optimized: Optional Numba and Multiprocessing.
# - Full Plotting: Generates all plots from the reference notebook.
#
# =============================================================================


# =============================================================================
# Section 1: Imports and Global Setup
# =============================================================================
import os
import glob
import yaml
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import mat73
from scipy.stats import chi2, f as finv
import itertools
from tqdm import tqdm
from scipy.signal import butter, filtfilt, iirnotch

# --- Optional Imports for Performance Enhancement ---
try:
    import multiprocessing
    MULTIPROCESSING_AVAILABLE = True
except ImportError:
    MULTIPROCESSING_AVAILABLE = False

try:
    from numba import njit, complex128, float64, int64, types
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# ==============================================================================
# Section 2: Configuration Management
# ==============================================================================

# This configuration is set up to EXACTLY replicate the notebook's analysis.
DEFAULT_CONFIG = {
    # --- Paths and Data ---
    'PATH_MAT_FILE': r"C:\PPGEE\Assessing CGST on ASSR\Numero_Deteccoes_consecutiva_H",
    'PATH_EEG_DATA': r"C:\Users\alexa\experimental_data\todos\ENTRADAS_PATRICIA",
    'BASE_OUTPUT_DIR': './plots/protocol_analysis/',

    # --- Performance Enhancements ---
    'USE_MULTIPROCESSING': True,
    'USE_NUMBA': True,

    # --- Core Analysis Parameters (Replication Settings) ---
    'VOLUNTEERS': ['Ab', 'An', 'Er', 'Qu', 'Sa', 'Ti', 'Wr'],
    'INTENSITY': '50dB',
    'MMAX': 240,
    'ALPHA': 0.05,
    'FP_DESEJADO': 0.05,
    'METRICS_TO_RUN': ['D-aMSC'], # FIXED: All metrics available
    'TARGET_ELECTRODES': ['Fz', 'Fcz', 'Cz', 'Pz', 'Oz', 'T5', 'T6'],

    # --- Preprocessing Pipeline (Replication Settings) ---
    'FILTER_TYPE': 'custom',
    'APPLY_NOTCH_FILTER': True,
    'ARTIFACT_REJECTION_TYPE': None,
    'ARTIFACT_THRESHOLD_VALUE': 0.1 / 200, # Matches notebook's 'remoc'
    'APPLY_MEAN_REMOVAL': True,
    'REMOVE_FIRST_TWO_WINDOWS': False,

    # --- Full list of available options (not all used in replica) ---
    'CUSTOM_FILTER_LOW_CUT': 30.0,
    'CUSTOM_FILTER_HIGH_CUT': 300.0,
    'CUSTOM_FILTER_ORDER': 2,
    'NOTCH_FREQ': 60.0,
    'NOTCH_Q': 100,
    'PERCENTILE_REMOVAL_AMOUNT': 15,

    # --- System Parameters ---
    'ALL_ELECTRODES': ['Fz','F3','F4','F7','Fcz','Cz','C3','C4','Pz','P3','P4','T3','T4','T5','T6','Oz']
}

def load_config_from_yaml(config_file):
    print(f"Loading configuration from: '{config_file}'")
    with open(config_file, 'r') as f:
        loaded_config = yaml.safe_load(f)
    config = DEFAULT_CONFIG.copy()
    config.update(loaded_config)
    return config

def save_config_to_yaml(config, run_output_dir):
    filepath = os.path.join(run_output_dir, "config.yml")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=4)
    print(f"Run configuration saved to: '{filepath}'\n")

# =============================================================================
# Section 3: Accelerator and Helper Function Definitions
# =============================================================================

# Global placeholders
msweep, dipolos_py, ETS_py, pareto_front_py = None, None, None, None
WORKER_INITIALIZED = False

def setup_accelerators(use_numba_flag):
    global msweep, dipolos_py, ETS_py, pareto_front_py
    # --- Standard Python Implementations (Fallback) ---
    def msweep_python(matrix, r):
        A = np.copy(matrix).astype(np.complex128)
        for k in range(r):
            d = A[k, k];
            if np.abs(d) < 1e-12: d = 1e-12
            col_k, row_k = A[:, k].copy(), A[k, :].copy()
            A -= np.outer(col_k, row_k) / d
            A[k, :], A[:, k], A[k, k] = row_k / d, -col_k / d, 1.0 / d
        return A

    def dipolos_python(x):
        n_samples, n_channels = x.shape
        n_combs = n_channels * (n_channels - 1) // 2
        y = np.zeros((n_samples, n_channels + n_combs))
        y[:, :n_channels] = x; idx = n_channels
        for i, j in itertools.combinations(range(n_channels), 2):
            y[:, idx] = x[:, i] - x[:, j]; idx += 1
        return y

    def ETS_python(ord_values, MM, alfa, NDC, vc_values):
        NDC = np.ceil(NDC); det = ord_values > vc_values
        consecutive_detections = 0
        for i, is_detected in enumerate(det):
            consecutive_detections = (consecutive_detections + 1) if is_detected else 0
            if consecutive_detections >= NDC: return 1, MM[i]
        return 0, MM[-1]

    def pareto_python(points):
        is_pareto = np.ones(points.shape[0], dtype=bool)
        for i, c in enumerate(points):
            if is_pareto[i]:
                is_pareto[is_pareto] = np.any(points[is_pareto] > c, axis=1) | np.all(points[is_pareto] == c, axis=1)
                is_pareto[i] = True
        return points[is_pareto], np.where(is_pareto)[0]

    # --- Numba JIT-Compiled Implementations (High-Performance) ---
    if use_numba_flag and NUMBA_AVAILABLE:
        @njit(complex128[:,:](complex128[:,:], int64), cache=True)
        def msweep_njit(matrix, r):
            A = np.copy(matrix)
            for k in range(r):
                d = A[k, k];
                if np.abs(d) < 1e-12: d = 1e-12
                col_k, row_k = A[:, k].copy(), A[k, :].copy()
                A -= np.outer(col_k, row_k) / d
                A[k, :], A[:, k], A[k, k] = row_k / d, -col_k / d, 1.0 / d
            return A

        @njit(float64[:,:](float64[:,:]), cache=True)
        def dipolos_njit(x):
            n_samples, n_channels = x.shape
            n_combs = n_channels * (n_channels - 1) // 2
            y = np.zeros((n_samples, n_channels + n_combs), dtype=np.float64)
            y[:, :n_channels] = x; idx = n_channels
            for i in range(n_channels):
                for j in range(i + 1, n_channels):
                    y[:, idx] = x[:, i] - x[:, j]; idx += 1
            return y

        @njit(types.UniTuple(int64, 2)(float64[:], int64[:], float64, float64, float64[:]), cache=True)
        def ETS_njit(ord_values, MM, alfa, NDC, vc_values):
            NDC = np.ceil(NDC); det = ord_values > vc_values
            consecutive_detections = 0
            for i, is_detected in enumerate(det):
                consecutive_detections = (consecutive_detections + 1) if is_detected else 0
                if consecutive_detections >= NDC: return 1, MM[i]
            return 0, MM[-1]

        @njit(types.Tuple((float64[:,:], types.int64[:]))(float64[:,:]), cache=True)
        def pareto_front_njit(points):
            n_points = points.shape[0]
            is_pareto = np.ones(n_points, dtype=np.bool_)
            for i in range(n_points):
                for j in range(n_points):
                    if i == j: continue
                    if np.all(points[j] >= points[i]) and np.any(points[j] > points[i]):
                        is_pareto[i] = False; break
            return points[np.where(is_pareto)[0]], np.where(is_pareto)[0]

        msweep, dipolos_py, ETS_py, pareto_front_py = msweep_njit, dipolos_njit, ETS_njit, pareto_front_njit
    else:
        msweep, dipolos_py, ETS_py, pareto_front_py = msweep_python, dipolos_python, ETS_python, pareto_python

# =============================================================================
# Section 4: Core Metric and Protocol Functions
# =============================================================================

# --- Standard Metric Calculations ---
def _calculate_univariate_msc(data, tj, fs, alpha):
    M = data.shape[0] // tj
    if M <= 1: return np.zeros(tj // 2 + 1), 1.0
    Y = np.fft.fft(data.flatten()[:M*tj].reshape(M, tj), axis=1)[:, :tj // 2 + 1]
    msc = np.zeros(Y.shape[1]); valid = np.sum(np.abs(Y)**2, axis=0) > 1e-12
    msc[valid] = np.abs(np.sum(Y, axis=0)[valid])**2 / (M * np.sum(np.abs(Y)**2, axis=0)[valid])
    return msc, (1 - alpha**(1/(M-1)) if M > 1 else 1.0)

def _calculate_univariate_csm(data, tj, fs, alpha):
    M = data.shape[0] // tj
    if M <= 1: return np.zeros(tj // 2 + 1), 1.0
    Y = np.fft.fft(data.flatten()[:M * tj].reshape(M, tj), axis=1)[:, :tj // 2 + 1]
    teta = np.angle(Y)
    sum_cos, sum_sin = np.sum(np.cos(teta), axis=0), np.sum(np.sin(teta), axis=0)
    return (sum_cos**2 + sum_sin**2)/(M**2), (chi2.ppf(1-alpha, 2)/(2*M) if M > 0 else 1.0)

def calculate_mmsc(data, tj, fs, alpha=0.05):
    n_samples, n_channels = data.shape if data.ndim > 1 else (data.shape[0], 1)
    if n_channels == 1: return _calculate_univariate_msc(data, tj, fs, alpha)
    M, nf = n_samples // tj, tj // 2 + 1
    if M <= n_channels: return np.zeros(nf), 1.0
    Sfft = np.fft.fft(data[:M*tj, :].T.reshape(n_channels, M, tj), axis=2)[:, :, :nf]
    mmsc = np.zeros(nf)
    for kf in range(nf):
        Sfft_slice = Sfft[:, :, kf]
        spec_a = np.zeros((n_channels + 1, n_channels + 1), dtype=np.complex128)
        spec_a[:n_channels, :n_channels] = Sfft_slice @ Sfft_slice.conj().T
        V = np.sum(Sfft_slice, axis=1)
        spec_a[n_channels, :n_channels], spec_a[:n_channels, n_channels] = V.conj(), V
        spec_a[n_channels, n_channels] = 1
        spec_as = msweep(spec_a, n_channels)
        mmsc[kf] = (1 - np.real(spec_as[n_channels, n_channels])) / M
    Fcrit = finv.ppf(1 - alpha, 2*n_channels, 2*(M-n_channels))
    crit = Fcrit / (((M-n_channels)/n_channels) + Fcrit) if M > n_channels else 1.0
    return mmsc, crit

def calculate_mcsm(data, tj, fs, alpha=0.05):
    n_samples, n_channels = data.shape if data.ndim > 1 else (data.shape[0], 1)
    if n_channels == 1: return _calculate_univariate_csm(data, tj, fs, alpha)
    M, nf = n_samples // tj, tj // 2 + 1
    if M == 0: return np.zeros(nf), 1.0
    Y = np.fft.fft(data[:M * tj, :].reshape(M, tj, n_channels), axis=1)[:, :nf, :]
    teta = np.angle(Y)
    C_mean = np.mean(np.cos(teta), axis=2); S_mean = np.mean(np.sin(teta), axis=2)
    teta_med = np.arctan2(S_mean, C_mean)
    sum_cos, sum_sin = np.sum(np.cos(teta_med), axis=0), np.sum(np.sin(teta_med), axis=0)
    return (sum_cos**2 + sum_sin**2)/(M**2), chi2.ppf(1-alpha, 2*n_channels)/(2*M*n_channels)

def calculate_d_amsc(data, tj, fs, alpha=0.05):
    if data.ndim == 1 or data.shape[1] < 2: return _calculate_univariate_msc(data, tj, fs, alpha)
    all_mscs = [_calculate_univariate_msc(data[:, i], tj, fs, alpha)[0] for i in range(data.shape[1])]
    M = data.shape[0] // tj
    return np.mean(np.array(all_mscs), axis=0), (1 - alpha**(1/(M-1)) if M > 1 else 1.0)

def calculate_d_acsm(data, tj, fs, alpha=0.05):
    if data.ndim == 1 or data.shape[1] < 2: return _calculate_univariate_csm(data, tj, fs, alpha)
    all_csms = [_calculate_univariate_csm(data[:, i], tj, fs, alpha)[0] for i in range(data.shape[1])]
    M = data.shape[0] // tj
    return np.mean(np.array(all_csms), axis=0), (chi2.ppf(1 - alpha, 2) / (2 * M) if M > 0 else 1.0)


# --- REPLICATION-SPECIFIC FUNCTIONS ---

def msc_fft_replica_py(Y):
    """Replication of the notebook's msc_fft.m function."""
    M = Y.shape[1]
    if M == 0: return np.full((Y.shape[0],), np.nan)
    numerator = np.abs(np.sum(Y, axis=1))**2
    denominator = M * np.sum(np.abs(Y)**2, axis=1)
    ord_val = np.zeros_like(numerator, dtype=float)
    non_zero_den_mask = denominator != 0
    ord_val[non_zero_den_mask] = numerator[non_zero_den_mask] / denominator[non_zero_den_mask]
    return ord_val

def VC_MSC_replica_py(M, alfa):
    """Replication of the notebook's VC_MSC.m function."""
    M_safe = np.asarray(M, dtype=float)
    invalid_mask = M_safe <= 1
    M_safe[invalid_mask] = np.nan
    return 1 - np.power(alfa, 1. / (M_safe - 1))

def ETS_replica_py(ord_values, MM, alfa, NDC):
    """Replication of the notebook's ETS.m function."""
    NDC = np.ceil(NDC)
    valor_critico = VC_MSC_replica_py(MM, alfa)
    det = ord_values[MM - 2] > valor_critico
    consecutive_detections = 0
    for i, is_detected in enumerate(det):
        consecutive_detections = (consecutive_detections + 1) if is_detected else 0
        if consecutive_detections >= NDC:
            return 1, MM[i]
    return 0, MM[-1]

# --- Protocol Functions ---

def protocolo_deteccao_replica_py(x, fs, parametros, metric_name):
    """
    Replication of the notebook's `protocolo_deteccao.m`, using the
    unconventional FFT-once method.
    """
    num_bins_protocol, num_windows_total = 120, x.shape[1]

    # NOTEBOOK's FFT METHOD: FFT is applied once across the window dimension (axis=1)
    # The input 'x' is expected to be (time_points, windows)
    xfft = np.fft.fft(x, axis=0)

    ord_matrix = np.zeros((num_bins_protocol, num_windows_total - 1))

    for M in range(2, num_windows_total + 1):
        xfft_slice = xfft[:num_bins_protocol, :M]
        ord_matrix[:, M - 2] = msc_fft_replica_py(xfft_slice)

    num_param_sets = parametros.shape[0]
    dr, time = np.zeros((num_bins_protocol, num_param_sets)), np.zeros((num_bins_protocol, num_param_sets))

    for ii in range(num_param_sets):
        Mmin, Mstep, Mmax, NDC, alfa_corr = parametros[ii, :]
        MM = np.arange(Mmin, Mmax + 1, Mstep, dtype=int)
        if MM.size == 0: continue

        for ll in range(num_bins_protocol):
            dr[ll, ii], time[ll, ii] = ETS_replica_py(ord_matrix[ll, :], MM, alfa_corr, NDC)

    return dr, time

def protocolo_deteccao_py(x, fs, parametros, metric_name):
    """Standard protocol for MMSC, MCSM, D-aMSC, etc."""
    num_bins_protocol, num_windows_total = 120, x.shape[1]
    time_pts = x.shape[0]
    # FIXED: Added all metrics to the dictionary to resolve key error
    METRIC_FUNCTIONS = {
        'MMSC': calculate_mmsc,
        'MCSM': calculate_mcsm,
        'D-aMSC': calculate_d_amsc,
        'D-aCSM': calculate_d_acsm
    }
    metric_func = METRIC_FUNCTIONS[metric_name]

    ord_matrix = np.zeros((num_bins_protocol, num_windows_total - 1))
    crit_matrix = np.zeros_like(ord_matrix)

    for M in range(2, num_windows_total + 1):
        # FIXED: Using robust reshaping that handles both 2D and 3D data
        data_cont = x[:, :M, :].transpose(1,0,2).reshape(-1, x.shape[2]) if x.ndim==3 else x[:,:M].T.reshape(-1, x.shape[-1])
        if data_cont.shape[0] >= time_pts:
            ord_vals, crit_val = metric_func(data_cont, time_pts, fs, alpha=0.05)
            ord_matrix[:, M - 2] = ord_vals[:num_bins_protocol]
            crit_matrix[:, M - 2] = crit_val

    num_param_sets = parametros.shape[0]
    dr, time = np.zeros((num_bins_protocol, num_param_sets)), np.zeros((num_bins_protocol, num_param_sets))
    for ii in range(num_param_sets):
        Mmin, Mstep, Mmax, NDC, alfa_corr = parametros[ii, :]
        MM = np.arange(Mmin, Mmax + 1, Mstep, dtype=int)
        if MM.size == 0 or np.any(MM-2 >= ord_matrix.shape[1]): continue

        vc_for_ets = crit_matrix[0, MM - 2]
        for ll in range(num_bins_protocol):
            dr[ll, ii], time[ll, ii] = ETS_py(ord_matrix[ll, MM - 2], MM, alfa_corr, NDC, vc_for_ets)

    return dr, time

# =============================================================================
# Section 5: Preprocessing and Main Execution Logic
# =============================================================================

def preprocess_data(data, fs, config):
    """Applies a configurable preprocessing pipeline."""
    eeg_windows = data.transpose(1, 0, 2)

    if config.get('REMOVE_FIRST_TWO_WINDOWS', False) and eeg_windows.shape[0] > 2:
        eeg_windows = eeg_windows[2:, :, :]

    if config.get('FILTER_TYPE') == 'custom':
        b, a = butter(config['CUSTOM_FILTER_ORDER'], [config['CUSTOM_FILTER_LOW_CUT'], config['CUSTOM_FILTER_HIGH_CUT']], 'bandpass', fs=fs)
        eeg_windows = filtfilt(b, a, eeg_windows, axis=1)

    if config.get('APPLY_NOTCH_FILTER', False):
        b_n, a_n = iirnotch(config['NOTCH_FREQ'], config['NOTCH_Q'], fs)
        eeg_windows = filtfilt(b_n, a_n, eeg_windows, axis=1)

    if config.get('APPLY_MEAN_REMOVAL', False) and config.get('CURRENT_METRIC') != 'MSC-REPLICA':
         eeg_windows -= np.mean(eeg_windows, axis=1, keepdims=True)

    rejection_type = config.get('ARTIFACT_REJECTION_TYPE')
    if rejection_type == 'fixed_threshold' and eeg_windows.shape[0] > 0:
        peak_abs = np.max(np.abs(eeg_windows), axis=(1, 2))
        threshold = config['ARTIFACT_THRESHOLD_VALUE']
        eeg_windows = eeg_windows[peak_abs <= threshold]
    elif rejection_type == 'percentile' and eeg_windows.shape[0] > 0:
        peak_abs = np.max(np.abs(eeg_windows), axis=(1, 2))
        threshold = np.percentile(peak_abs, 100 - config['PERCENTILE_REMOVAL_AMOUNT'])
        eeg_windows = eeg_windows[peak_abs <= threshold]


    return eeg_windows.transpose(1, 0, 2)


def process_volunteer(args):
    """Main worker function for processing a single volunteer."""
    global WORKER_INITIALIZED
    voluntario_code, config = args

    if not WORKER_INITIALIZED:
        setup_accelerators(config.get('USE_NUMBA', False))
        WORKER_INITIALIZED = True

    metric = config['CURRENT_METRIC']
    try:
        eeg_filepath = os.path.join(config['PATH_EEG_DATA'], f"{voluntario_code}{config['INTENSITY']}.mat")
        eeg_data = mat73.loadmat(eeg_filepath)
        x_all_ele, Fs, binsM = eeg_data['x'], float(eeg_data['Fs']), (np.array(eeg_data['binsM']).flatten() - 1).astype(int)

        # --- REPLICATION-SPECIFIC PREPROCESSING ---
        if metric == 'MSC-REPLICA':
            x_subset = x_all_ele[:, :, config['ELECTRODE_INDICES'][0]] # Replica is single-electrode
            # 1. Mean removal across windows (notebook's method)
            if config.get('APPLY_MEAN_REMOVAL', False):
                 x_subset = x_subset - x_subset.mean(axis=0, keepdims=True)

            # 2. Transpose for window-based operations
            x_subset = x_subset.transpose(1, 0)[:, :, np.newaxis] # Shape: (windows, time, 1)

            # 3. Discard first two windows
            if config.get('REMOVE_FIRST_TWO_WINDOWS', False):
                x_subset = x_subset[2:, :, :]

            # 4. Artifact rejection
            rejection_type = config.get('ARTIFACT_REJECTION_TYPE')
            if rejection_type == 'fixed_threshold' and x_subset.shape[0] > 0:
                peak_abs = np.max(np.abs(x_subset), axis=(1, 2))
                threshold = config['ARTIFACT_THRESHOLD_VALUE']
                x_subset = x_subset[peak_abs <= threshold]

            # 5. Final formatting for replica protocol
            clean_data = x_subset.transpose(1, 0, 2).squeeze() # Shape: (time, windows)
            clean_data = clean_data[:, :config['MMAX']]

            if clean_data.shape[1] < 2: return None

            dr, time = protocolo_deteccao_replica_py(clean_data, Fs, config['PARAMETROS'], metric)

        # --- STANDARD (MMSC, D-aMSC, etc.) PREPROCESSING ---
        else:
            x_multi_ele = x_all_ele[:, :, config['ELECTRODE_INDICES']]
            processed_data = preprocess_data(x_multi_ele, Fs, config)

            # FIXED: Added logic to create dipoles for D-aMSC and D-aCSM
            if metric in ['D-aMSC', 'D-aCSM']:
                # For dipole metrics, we need to handle the window dimension separately
                if processed_data.shape[1] > 0:
                    input_data = np.array([dipolos_py(processed_data[:, i, :]) for i in range(processed_data.shape[1])]).transpose(1,0,2)
                else:
                    input_data = processed_data
            else:
                input_data = processed_data

            clean_data = input_data[:, :config['MMAX']]
            if clean_data.shape[1] < 2: return None

            dr, time = protocolo_deteccao_py(clean_data, Fs, config['PARAMETROS'], metric)

        return dr, time, binsM
    except Exception as e:
        print(f"Warning: Failed to process {voluntario_code} for metric {metric}. Error: {e}")
        return None

def main(config):
    setup_accelerators(config['USE_NUMBA'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(config['BASE_OUTPUT_DIR'], f"run_{timestamp}")
    save_config_to_yaml(config, run_output_dir)

    mat_filename = f"NDC_AlfaCorrigido_Mmax{config['MMAX']}_alfa_{config['ALPHA']}_FPdesejado{config['FP_DESEJADO']}.mat"
    mat_filepath = os.path.join(config['PATH_MAT_FILE'], mat_filename)
    try:
        mat_contents = mat73.loadmat(mat_filepath)
        P, alfa_corrigido, NDC_minimo = mat_contents['P'], mat_contents['alfa_corrigido'], mat_contents['NDC_minimo']
    except Exception as e:
        raise FileNotFoundError(f"Could not load/parse parameter file: {mat_filepath}\nError: {e}")

    config['PARAMETROS'] = np.hstack((P, np.array(NDC_minimo).reshape(-1, 1), np.array(alfa_corrigido).reshape(-1, 1)))
    config['ELECTRODE_INDICES'] = [config['ALL_ELECTRODES'].index(e) for e in config['TARGET_ELECTRODES']]

    results = {}
    for metric in config['METRICS_TO_RUN']:
        print(f"\n===== Running analysis for metric: {metric} =====")
        config['CURRENT_METRIC'] = metric
        tasks = [(vol, config) for vol in config['VOLUNTEERS']]

        Tdr_all, Ttime_all, binsM = [], [], None

        if config['USE_MULTIPROCESSING'] and MULTIPROCESSING_AVAILABLE:
            with multiprocessing.Pool() as pool:
                pool_results = list(tqdm(pool.imap(process_volunteer, tasks), total=len(tasks), desc=f"Volunteers ({metric})"))
        else:
            pool_results = [process_volunteer(task) for task in tqdm(tasks, desc=f"Volunteers ({metric})")]

        for res in pool_results:
            if res:
                dr, time, bM = res
                Tdr_all.append(dr); Ttime_all.append(time)
                if binsM is None: binsM = bM

        if not Tdr_all:
            print(f"Warning: No valid data for metric {metric} after processing all volunteers.")
            continue

        Tdr = np.stack(Tdr_all, axis=-1)
        Ttime = np.stack(Ttime_all, axis=-1)

        TXD = np.mean(Tdr[binsM, ...], axis=(0, 2)) * 100
        binsR = np.setdiff1d(np.arange(100), binsM); binsR = binsR[binsR > 1]
        FP = np.mean(Tdr[binsR, ...], axis=(0, 2)) * 100

        timeM = Ttime[binsM, :, :]; timeM[timeM == -1] = config['MMAX']
        timeM_mean = np.mean(timeM, axis=(0, 2))

        results[metric] = {'txd': TXD, 'fpr': FP, 'time': timeM_mean, 'params': config['PARAMETROS']}

    # --- Plotting Section ---
    print("\n--- Plotting Combined Results ---")
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    # Plot 1: Combined Pareto Front
    fig1, ax1 = plt.subplots(figsize=(14, 9))
    for i, (metric, data) in enumerate(results.items()):
        points = np.vstack((data['txd'], -data['time'])).T
        p_front, _ = pareto_front_py(points)
        p_front = p_front[np.argsort(p_front[:, 0])]
        ax1.plot(p_front[:, 0], -p_front[:, 1], 'o-', color=colors[i], label=f'{metric} Pareto Front')

    ax1.set_title(f"Pareto Front: Detection Rate vs. Exam Time ({config['INTENSITY']})", fontsize=16)
    ax1.set_xlabel('True Positive Rate (TPR, %)', fontsize=12)
    ax1.set_ylabel('Mean Exam Time (s)', fontsize=12)
    ax1.grid(True, linestyle='--'); ax1.legend(); ax1.set_ylim(bottom=0); ax1.set_xlim(left=0, right=100)
    plt.tight_layout(); plt.savefig(os.path.join(run_output_dir, "pareto_tpr_vs_time_combined.png"))
    plt.show()

    # Plot 2: Combined FPR vs. Time
    fig2, ax2 = plt.subplots(figsize=(14, 9))
    for i, (metric, data) in enumerate(results.items()):
        sorted_indices = np.argsort(data['time'])
        ax2.plot(data['time'][sorted_indices], data['fpr'][sorted_indices], 'o-', color=colors[i], label=metric, alpha=0.8)

    ax2.set_title(f"Performance: False Positive Rate vs. Exam Time ({config['INTENSITY']})", fontsize=16)
    ax2.set_xlabel('Mean Exam Time (s)', fontsize=12)
    ax2.set_ylabel('False Positive Rate (FPR, %)', fontsize=12)
    ax2.grid(True, linestyle='--'); ax2.legend(); ax2.set_ylim(bottom=0); ax2.set_xlim(left=0)
    plt.tight_layout(); plt.savefig(os.path.join(run_output_dir, "fpr_vs_time_combined.png"))
    plt.show()

    print("\nAnalysis complete.")


# =============================================================================
# Section 6: Entry Point
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Executes the EEG protocol analysis pipeline.")
    parser.add_argument('--config', type=str, help='Path to the configuration .yml file.')
    args = parser.parse_args()

    config = load_config_from_yaml(args.config) if args.config and os.path.exists(args.config) else DEFAULT_CONFIG

    if config.get('USE_MULTIPROCESSING', False):
        multiprocessing.freeze_support()

    main(config)