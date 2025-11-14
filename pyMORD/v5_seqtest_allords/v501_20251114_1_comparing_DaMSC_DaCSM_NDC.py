import numpy as np
from scipy.optimize import minimize
from scipy.fft import fft
import matplotlib.pyplot as plt
from numba import jit
import concurrent.futures
from tqdm import tqdm
import h5py      # For reading .mat v7.3 files
import os        # For handling file paths
from datetime import datetime
import shutil    # For file operations if needed
import mat73     # To load modern .mat files
from scipy.signal import butter, filtfilt, iirnotch # For preprocessing filters
import pandas as pd # For the robust Pareto front plotting
import matplotlib.colors as mcolors # For creating colormaps
import time      # For timing analysis
import sys       # For logging output
from scipy.stats import chi2 # For CSM critical value calculation

# --- Global Path for Data ---
# Set the path to the directory containing your .mat files
PATH_EEG_DATA = r"C:\Users\alexa\experimental_data\todos\ENTRADAS_PATRICIA"

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
class Logger(object):
    """A simple logger to write console output to a file."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ==============================================================================
# METRIC DEFINITIONS
# ==============================================================================
@jit(nopython=True)
def msc_fft(Y, M):
    """Calculates Magnitude-Squared Coherence from FFT data."""
    if Y.shape[1] != M:
        raise ValueError("Window size mismatch")
    denominator = M * np.sum(np.abs(Y)**2, axis=1)
    return np.abs(np.sum(Y, axis=1))**2 / (denominator + 1e-12)

@jit(nopython=True)
def csm_fft(Y, M):
    """Calculates Coherent Phase Statistic from FFT data."""
    if Y.shape[1] != M:
        raise ValueError("Window size mismatch")
    teta = np.angle(Y)
    sum_cos = np.zeros(teta.shape[0])
    sum_sin = np.zeros(teta.shape[0])
    for i in range(teta.shape[0]):
        for j in range(teta.shape[1]):
            sum_cos[i] += np.cos(teta[i, j])
            sum_sin[i] += np.sin(teta[i, j])
    return (sum_cos**2 + sum_sin**2) / (M**2)

@jit(nopython=True)
def vc_msc(M, alfa):
    """Critical value for MSC."""
    # This function expects M as a vector/array for compatibility
    result = np.zeros_like(M, dtype=np.float64)
    for i in range(len(M)):
        if M[i] > 1:
            result[i] = 1 - alfa**(1 / (M[i] - 1))
    return result

def vc_csm(M, alfa):
    """Critical value for CSM."""
    # Ensure M is a numpy array for vectorized operations
    M_arr = np.array(M)
    # Avoid division by zero for M=0 case, although unlikely for M>=2 protocols
    crit = np.zeros_like(M_arr, dtype=float)
    valid_M = M_arr > 0
    crit[valid_M] = chi2.ppf(1 - alfa, 2) / (2 * M_arr[valid_M])
    return crit

METRIC_HANDLERS = {
    'MSC': {'calculator': msc_fft, 'critical_value_func': vc_msc},
    'CSM': {'calculator': csm_fft, 'critical_value_func': vc_csm}
}

# ==============================================================================
# H0 SIMULATION GENERATORS (Updated for Metric Flexibility)
# ==============================================================================
def generate_h0_baseline(nRuns, Mmax, tj, bin_freq, metric_calculator):
    print(f"\n[{datetime.now()}] Generating H0 data using 'Baseline' method...")
    ord_sim = np.zeros((nRuns, Mmax + 1))
    for ii in tqdm(range(nRuns), desc="Simulating H0 (Baseline)"):
        x_reshaped = np.random.randn(tj, Mmax)
        xfft = fft(x_reshaped, axis=0)
        for M in range(2, Mmax + 1):
            metric_results = metric_calculator(xfft[:, :M], M)
            ord_sim[ii, M] = metric_results[bin_freq]
    return ord_sim

def generate_h0_variable_snr(nRuns, Mmax, tj, bin_freq, noise_mean_db, noise_std_db, metric_calculator):
    print(f"\n[{datetime.now()}] Generating H0 data using 'Variable SNR' method (Mean={noise_mean_db}dB, Std={noise_std_db}dB)...")
    ord_sim = np.zeros((nRuns, Mmax + 1))
    for ii in tqdm(range(nRuns), desc="Simulating H0 (Variable SNR)"):
        noise_powers_db = noise_mean_db + noise_std_db * np.random.randn(Mmax)
        noise_powers_linear = 10**(noise_powers_db / 10)
        noise_samples = np.random.randn(tj, Mmax) * np.sqrt(noise_powers_linear)
        xfft = fft(noise_samples, axis=0)
        for M in range(2, Mmax + 1):
            metric_results = metric_calculator(xfft[:, :M], M)
            ord_sim[ii, M] = metric_results[bin_freq]
    return ord_sim

def generate_h0_hybrid_snr(nRuns, Mmax, tj, bin_freq, snr_levels_db, metric_calculator):
    print(f"\n[{datetime.now()}] Generating H0 data using 'Hybrid SNR' method...")
    num_pools = len(snr_levels_db)
    runs_per_pool = int(np.ceil(nRuns * 2 / num_pools))
    pooled_ord_results = []
    for snr_db in snr_levels_db:
        ord_pool_run = np.zeros((runs_per_pool, Mmax + 1))
        noise_power_linear = 10**(snr_db / 10)
        for ii in range(runs_per_pool):
            noise_samples = np.random.randn(tj, Mmax) * np.sqrt(noise_power_linear)
            xfft = fft(noise_samples, axis=0)
            for M in range(2, Mmax + 1):
                metric_results = metric_calculator(xfft[:, :M], M)
                ord_pool_run[ii, M] = metric_results[bin_freq]
        pooled_ord_results.append(ord_pool_run)
    full_pool = np.concatenate(pooled_ord_results, axis=0)
    final_indices = np.random.choice(full_pool.shape[0], nRuns, replace=False)
    return full_pool[final_indices, :]

def generate_h0_hybrid_expnsim(nRuns, Mmax, tj, bin_freq, shared_config, preprocess_sim_data, metric_calculator):
    print(f"\n[{datetime.now()}] Generating H0 data using 'Hybrid EXPnSIM' method...")
    experimental_ord_pool = []
    for vol in tqdm(shared_config['Vvoluntario'], desc="Loading Experimental Data"):
        try:
            filepath = os.path.join(shared_config['caminho'], f"{vol}{shared_config['Intensidade'][0]}.mat")
            if not os.path.exists(filepath): continue
            eeg_data = mat73.loadmat(filepath)
            x_raw = eeg_data['x'][:, :, shared_config['pos_ele'] - 1]
            Fs = float(eeg_data['Fs'])
            binsM_from_file = (np.array(eeg_data['binsM']).flatten() - 1).astype(int)
            noise_bins = np.setdiff1d(np.arange(shared_config['binsM_count']), binsM_from_file)
            x_clean_3d = preprocess_data(x_raw.reshape(x_raw.shape[0], x_raw.shape[1], 1), Fs, shared_config)
            x_clean = x_clean_3d[:, :Mmax, 0]
            if x_clean.shape[1] < 2: continue
            xfft = fft(x_clean, axis=0)
            num_bins_calculated = x_clean.shape[0]
            for noise_bin_idx in noise_bins:
                if noise_bin_idx >= num_bins_calculated: continue
                ord_run = np.zeros(Mmax + 1)
                for M in range(2, x_clean.shape[1] + 1):
                    metric_results = metric_calculator(xfft[:, :M], M)
                    ord_run[M] = metric_results[noise_bin_idx]
                experimental_ord_pool.append(ord_run)
        except Exception as e:
            print(f"Warning: Could not process volunteer {vol}. Error: {e}")
    if not experimental_ord_pool:
        raise RuntimeError("Failed to extract valid noise from experimental data.")
    simulated_ord_pool = np.zeros((nRuns, Mmax + 1))
    for ii in tqdm(range(nRuns), desc="Simulating H0 (for Hybrid)"):
        sim_noise = np.random.randn(tj, Mmax)
        if preprocess_sim_data:
            sim_noise_3d = sim_noise.reshape(tj, Mmax, 1)
            processed_sim_3d = preprocess_data(sim_noise_3d, shared_config['Fs'], shared_config)
            sim_noise = processed_sim_3d[:, :, 0]
        xfft_sim = fft(sim_noise, axis=0)
        for M in range(2, sim_noise.shape[1] + 1):
            metric_results = metric_calculator(xfft_sim[:, :M], M)
            simulated_ord_pool[ii, M] = metric_results[bin_freq]
    full_pool = np.vstack([np.array(experimental_ord_pool), simulated_ord_pool])
    final_indices = np.random.choice(full_pool.shape[0], nRuns, replace=False)
    return full_pool[final_indices, :]

def generate_h0_hybrid_estimated_params(nRuns, Mmax, tj, bin_freq, shared_config, metric_calculator):
    print(f"\n[{datetime.now()}] Generating H0 data using 'Hybrid Estimated Params' method...")
    all_noise_window_powers = []
    print("  Estimating noise power distribution from experimental data...")
    for vol in tqdm(shared_config['Vvoluntario'], desc="Analyzing Experimental Noise"):
        try:
            filepath = os.path.join(shared_config['caminho'], f"{vol}{shared_config['Intensidade'][0]}.mat")
            if not os.path.exists(filepath): continue
            eeg_data = mat73.loadmat(filepath)
            x_raw = eeg_data['x'][:, :, shared_config['pos_ele'] - 1]
            Fs = float(eeg_data['Fs'])
            x_clean_3d = preprocess_data(x_raw.reshape(x_raw.shape[0], x_raw.shape[1], 1), Fs, shared_config)
            x_clean = x_clean_3d[:, :Mmax, 0]
            window_powers = np.var(x_clean, axis=0)
            all_noise_window_powers.extend(window_powers)
        except Exception as e:
            print(f"Warning: Could not process volunteer {vol} for param estimation. Error: {e}")
    if not all_noise_window_powers:
        raise RuntimeError("Failed to estimate noise parameters from experimental data.")
    powers_db = 10 * np.log10(np.array(all_noise_window_powers))
    estimated_mean_db = np.nanmean(powers_db)
    estimated_std_db = np.nanstd(powers_db)
    print(f"  Estimated Noise Parameters: Mean = {estimated_mean_db:.2f} dB, Std Dev = {estimated_std_db:.2f} dB")
    return generate_h0_variable_snr(nRuns, Mmax, tj, bin_freq,
                                    noise_mean_db=estimated_mean_db,
                                    noise_std_db=estimated_std_db,
                                    metric_calculator=metric_calculator)

# =============================================================================
# Section: Preprocessing
# =============================================================================
def preprocess_data(data, fs, config):
    eeg_windows = data.transpose(1, 0, 2)
    if config.get('FILTER_TYPE') == 'custom':
        b, a = butter(config['CUSTOM_FILTER_ORDER'],
                      [config['CUSTOM_FILTER_LOW_CUT'], config['CUSTOM_FILTER_HIGH_CUT']],
                      'bandpass', fs=fs)
        eeg_windows = filtfilt(b, a, eeg_windows, axis=1)
    if config.get('APPLY_NOTCH_FILTER', False):
        b_n, a_n = iirnotch(config['NOTCH_FREQ'], config['NOTCH_Q'], fs)
        eeg_windows = filtfilt(b_n, a_n, eeg_windows, axis=1)
    if config.get('APPLY_MEAN_REMOVAL', False):
        eeg_windows -= np.mean(eeg_windows, axis=1, keepdims=True)
    if config.get('ARTIFACT_REJECTION_TYPE') == 'fixed_threshold' and eeg_windows.shape[0] > 0:
        peak_abs = np.max(np.abs(eeg_windows), axis=(1, 2))
        eeg_windows = eeg_windows[peak_abs <= config['ARTIFACT_THRESHOLD_VALUE']]
    return eeg_windows.transpose(1, 0, 2)

# ==============================================================================
# CORE ANALYSIS FUNCTIONS (Updated for Metric Flexibility)
# ==============================================================================
@jit(nopython=True)
def ETS(ord_values_for_protocol, MM, valor_critico, NDC):
    NDC = int(np.ceil(NDC))
    det = ord_values_for_protocol > valor_critico
    consecutive_detections = 0
    for i in range(len(MM)):
        if det[i]:
            consecutive_detections += 1
        else:
            consecutive_detections = 0
        if consecutive_detections == NDC:
            return 1, MM[i]
    return 0, MM[-1]

def funcao_custo_v2(alfa_array, NDC, MM, ord_0idx, FP_desejado, critical_value_func):
    alfa = alfa_array[0]
    alfa = max(1e-9, min(alfa, 1.0 - 1e-9))
    nRuns = ord_0idx.shape[0]
    dr = np.zeros(nRuns)
    valor_critico = critical_value_func(MM, alfa)
    protocol_indices = MM - 1 # -2
    for ii in range(nRuns):
        ord_values_for_protocol = ord_0idx[ii, protocol_indices]
        detection_result, _ = ETS(ord_values_for_protocol, MM, valor_critico, NDC)
        dr[ii] = detection_result
    FP = np.mean(dr)
    erro = FP - FP_desejado
    J = 0.5 * erro**2
    return J, np.array([erro])

def estimarNDC(NDCinicial, alfa, FP_desejado, ord_0idx, Mmin, Mstep, Mmax, critical_value_func):
    MM = np.arange(Mmin, Mmax, Mstep)
    protocol_indices = MM - 1 # -2
    nRuns = ord_0idx.shape[0]
    FP_list = []
    valor_critico = critical_value_func(MM, alfa)
    for NDC in range(NDCinicial, len(MM)):
        dr = np.zeros(nRuns)
        for ii in range(nRuns):
            # print(protocol_indices)
            ord_values_for_protocol = ord_0idx[ii, protocol_indices]
            dr[ii], _ = ETS(ord_values_for_protocol, MM, valor_critico, NDC)
        FP = np.mean(dr)
        FP_list.append(FP)
        if FP <= FP_desejado:
            break
    FP_array = np.array(FP_list)
    ind = np.where((FP_array < FP_desejado) & (FP_array != 0))[0]
    if not ind.any():
        return np.nan, FP_array
    elif ind[0] > 0:
        return np.interp(FP_desejado, [FP_array[ind[0]], FP_array[ind[0]-1]], [ind[0]+1, ind[0]]), FP_array
    else:
        return 1, FP_array

def parametros_protocolo(Mmax):
    P_list = []
    for m_step in range(1, Mmax):
        for m_min in range(2, Mmax):
            k = (Mmax - m_min) / m_step
            if k >= 0 and k % 1 == 0:
                P_list.append([m_min, m_step, Mmax])
    P_list.append([Mmax, 1, Mmax])
    return np.array(P_list)

def funcao_NDC_alfaCorrigido_Mmax(ord_sim, Mmax, alfa_teste, FP_desejado, critical_value_func):
    nRuns = ord_sim.shape[0]
    ord_sim_0idx = ord_sim[:, 2:]
    P = parametros_protocolo(Mmax)
    print(f"Generated {P.shape[0]} protocol parameters to optimize.")
    alfa_corrigido = np.full(P.shape[0], np.nan)
    NDC_minimo = np.full(P.shape[0], np.nan)
    sort_metric = np.ceil((Mmax - P[:, 0]) / P[:, 1]) + 1
    sort_indices = np.argsort(sort_metric)
    P = P[sort_indices]
    for ii in tqdm(range(P.shape[0]), desc="Optimizing Protocol Parameters"):
        Mmin, Mstep, Mmax_local = P[ii, 0], P[ii, 1], P[ii, 2]
        NDC, _ = estimarNDC(1, alfa_teste, FP_desejado, ord_sim_0idx, Mmin, Mstep, Mmax_local, critical_value_func)
        if np.isnan(NDC): continue
        NDC_minimo[ii] = NDC
        MM = np.arange(Mmin, Mmax_local + 1, Mstep)
        res = minimize(funcao_custo_v2, alfa_teste, args=(NDC, MM, ord_sim_0idx, FP_desejado, critical_value_func),
                       method='CG', jac=True, options={'maxiter': 100, 'gtol':0.5*(1/88)**2})
        if res.success:
            alfa_corrigido[ii] = res.x[0]
    return alfa_corrigido, NDC_minimo, None, P

def protocolo_deteccao(x, parametros, binsM_count, metric_calculator, critical_value_func):
    n_samples, n_windows = x.shape
    n_freq_bins = binsM_count
    ord_val_0idx = np.zeros((n_freq_bins, n_windows - 1))
    xfft = fft(x, axis=0)
    for M in range(2, n_windows + 1):
        metric_results = metric_calculator(xfft[:, :M], M)
        ord_val_0idx[:len(metric_results), M - 2] = metric_results[:n_freq_bins]
    dr = np.zeros((n_freq_bins, len(parametros)))
    time_val = np.zeros((n_freq_bins, len(parametros)))
    for i, p in enumerate(parametros):
        Mmin, Mstep, Mmax, NDC, alfa = int(p[0]), int(p[1]), int(p[2]), p[3], p[4]
        if Mmin >= Mmax: continue
        MM = np.arange(Mmin, Mmax + 1, Mstep)
        valor_critico = critical_value_func(MM, alfa)
        protocol_indices = MM # - 1 # -2
        if np.any(protocol_indices >= ord_val_0idx.shape[1]): continue
        for ll in range(n_freq_bins):
            ord_values_for_protocol = ord_val_0idx[ll, protocol_indices]
            dr[ll, i], time_val[ll, i] = ETS(ord_values_for_protocol, MM, valor_critico, NDC)
    return dr, time_val

# ==============================================================================
# VOLUNTEER PROCESSING AND MAIN ORCHESTRATOR
# ==============================================================================
def process_volunteer(args):
    cont_vol, Vvoluntario, Intensidade, Mmax, parametros, caminho, pos_ele, _, binsM_count, config_for_process, metric_calculator, critical_value_func = args
    voluntario = Vvoluntario[cont_vol]
    intensidade = Intensidade[0]
    filepath = os.path.join(caminho, f"{voluntario}{intensidade}.mat")
    if not os.path.exists(filepath):
        return None, None
    try:
        eeg_data = mat73.loadmat(filepath)
        x_data = eeg_data['x']
        Fs = float(eeg_data.get('Fs', 1000.0))
    except Exception:
        with h5py.File(filepath, 'r') as f:
            x_data = f['x'][:]
            Fs = 1000.0
    x_single_electrode = x_data[:, :, pos_ele - 1]
    x_reshaped = x_single_electrode.reshape(x_single_electrode.shape[0], x_single_electrode.shape[1], 1)
    x_clean_3d = preprocess_data(x_reshaped, Fs, config_for_process)
    x_clean_2d = x_clean_3d[:, :Mmax, 0]
    if x_clean_2d.shape[1] < 2:
        return None, None
    return protocolo_deteccao(x_clean_2d, parametros, binsM_count, metric_calculator, critical_value_func)

def paretoFront(points):
    num_points = points.shape[0]
    is_dominated = np.zeros(num_points, dtype=bool)
    for i in range(num_points):
        for j in range(num_points):
            if i == j: continue
            if np.all(points[j] >= points[i]) and np.any(points[j] > points[i]):
                is_dominated[i] = True
                break
    return points[~is_dominated], np.where(~is_dominated)[0]

def save_results(filepath, ord_sim, parametros, config):
    np.savez(filepath, ord_sim=ord_sim, parametros=parametros, config=config)
    print(f"Results saved to '{filepath}'")

def load_results(filepath):
    print(f"Loading cached results from '{filepath}'...")
    data = np.load(filepath, allow_pickle=True)
    return data['ord_sim'], data['parametros'], data['config'].item()

def main():
    master_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_output_dir = os.path.join('./plots/pareto_comparisons/', f"run_{master_timestamp}")
    os.makedirs(master_output_dir, exist_ok=True)
    print(f"Master output directory created at: '{master_output_dir}'")

    RECOMPUTE_H0_OPTIMIZATION = False

    base_config = {
        'caminho': PATH_EEG_DATA,
        'Vvoluntario': ['Ab', 'An', 'Bb', 'Er', 'Lu', 'So', 'Qu', 'Vi', 'Sa', 'Ti', 'Wr'],
        'nRuns': 1000, 'alfa_teste': 0.05,
        'FP_desejado': 0.05, 'pos_ele': 1, 'binsM_count': 100, #fala que sao 120 mas no codigo tem 100.
          'tj': 32,
        'bin_freq': 8, 'Fs': 1000.0, 'FILTER_TYPE': None,
        'APPLY_NOTCH_FILTER': False, 'APPLY_MEAN_REMOVAL': True,
        'ARTIFACT_REJECTION_TYPE': False, 'ARTIFACT_THRESHOLD_VALUE': 0.1 / 200,
        'CUSTOM_FILTER_LOW_CUT': 30.0, 'CUSTOM_FILTER_HIGH_CUT': 300.0,
        'CUSTOM_FILTER_ORDER': 2, 'NOTCH_FREQ': 60.0, 'NOTCH_Q': 100,
        # --- New configurable parameters ---
        #  '70dB': 50, '60dB': 50, '50dB': 240, '40dB': 440, '30dB': 440, 'ESP': 20
        'INTENSITIES_TO_ANALYZE': {
           '50dB': 240,
        },
        'METRICS_TO_ANALYZE': ['MSC', 'CSM'],
    }

    SIMULATION_CONFIGS = []
    SIMULATION_CONFIGS.append({'name': 'Baseline_H0', 'generator': generate_h0_baseline, 'params': {}})
    SIMULATION_CONFIGS.append({'name': 'Hybrid_SNR_H0', 'generator': generate_h0_hybrid_snr, 'params': {'snr_levels_db': [-20,-15,-10]}})
    noise_means_db = [-15]
    noise_stds_db = [5]
    for mean_db in noise_means_db:
        for std_db in noise_stds_db:
            SIMULATION_CONFIGS.append({
                'name': f'VarSNR_Mean_{mean_db}dB_Std_{std_db}dB',
                'generator': generate_h0_variable_snr,
                'params': {'noise_mean_db': mean_db, 'noise_std_db': std_db}
            })
    # SIMULATION_CONFIGS.append({'name': 'Hybrid_EstimatedParams_H0', 'generator': generate_h0_hybrid_estimated_params, 'params': {}})
    # SIMULATION_CONFIGS.append({'name': 'Hybrid_EXPnSIM_H0', 'generator': generate_h0_hybrid_expnsim, 'params': {'preprocess_sim_data': True}})

    for intensity, mmax in base_config['INTENSITIES_TO_ANALYZE'].items():
        intensity_dir = os.path.join(master_output_dir, intensity)
        os.makedirs(intensity_dir, exist_ok=True)
        
        log_file_path = os.path.join(intensity_dir, 'run.log')
        original_stdout = sys.stdout
        sys.stdout = Logger(log_file_path)
        
        print(f"\n{'='*80}")
        print(f"--- PROCESSING INTENSITY: {intensity} | Mmax: {mmax} ---")
        print(f"{'='*80}\n")
        
        shared_config = base_config.copy()
        shared_config['Intensidade'] = [intensity]
        shared_config['Mmax'] = mmax

        all_run_dfs_for_intensity = {}

        for metric_name in shared_config['METRICS_TO_ANALYZE']:
            print(f"\n--- STARTING ANALYSIS FOR METRIC: {metric_name} ---\n")
            metric_handler = METRIC_HANDLERS[metric_name]
            metric_calculator = metric_handler['calculator']
            critical_value_func = metric_handler['critical_value_func']

            for run_config in SIMULATION_CONFIGS:
                run_name_with_metric = f"{run_config['name']}_{metric_name}"
                print(f"\n\n--- STARTING SIMULATION: {run_name_with_metric} for {intensity} ---\n")
                
                results_path = os.path.join(intensity_dir, f"{run_name_with_metric}_computed_data.npz")
                timings = {}

                if not RECOMPUTE_H0_OPTIMIZATION and os.path.exists(results_path):
                    ord_sim, parametros, _ = load_results(results_path)
                else:
                    t0 = time.time()
                    call_params = run_config['params'].copy()
                    if run_config['generator'] in [generate_h0_hybrid_estimated_params, generate_h0_hybrid_expnsim]:
                        call_params['shared_config'] = shared_config
                    
                    ord_sim = run_config['generator'](
                        nRuns=shared_config['nRuns'], Mmax=shared_config['Mmax'], 
                        tj=shared_config['tj'], bin_freq=shared_config['bin_freq'], 
                        metric_calculator=metric_calculator, **call_params)
                    timings['h0_simulation_s'] = time.time() - t0
                    
                    t0 = time.time()
                    alfa_corrigido, NDC_minimo, _, P = funcao_NDC_alfaCorrigido_Mmax(
                        ord_sim, Mmax=shared_config['Mmax'], 
                        alfa_teste=shared_config['alfa_teste'], 
                        FP_desejado=shared_config['FP_desejado'],
                        critical_value_func=critical_value_func)
                    timings['param_optimization_s'] = time.time() - t0
                    
                    parametros = np.hstack([P, NDC_minimo[:, np.newaxis], alfa_corrigido[:, np.newaxis]])
                    save_results(results_path, ord_sim, parametros, shared_config)
                    parametros = parametros[~np.isnan(parametros).any(axis=1)]

                if parametros.shape[0] == 0:
                    print(f"Optimization failed for {run_name_with_metric} at {intensity}. Skipping analysis.")
                    continue

                t0 = time.time()
                args_list = [(i, shared_config['Vvoluntario'], shared_config['Intensidade'], 
                              shared_config['Mmax'], parametros, shared_config['caminho'], 
                              shared_config['pos_ele'], shared_config['ARTIFACT_THRESHOLD_VALUE'], 
                              shared_config['binsM_count'], shared_config,
                              metric_calculator, critical_value_func) for i in range(len(shared_config['Vvoluntario']))]
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    results = list(tqdm(executor.map(process_volunteer, args_list), total=len(args_list), desc=f"Processing Volunteers ({run_name_with_metric} @ {intensity})"))
                timings['volunteer_processing_s'] = time.time() - t0
                
                with open(os.path.join(intensity_dir, f"{run_name_with_metric}_timing_log.txt"), "w") as f:
                    for key, val in timings.items():
                        f.write(f"{key}: {val:.2f}\n")
                
                Tdr_list, Ttime_list = [], []
                for dr, time_val in results:
                    if dr is not None and time_val is not None:
                        Tdr_list.append(dr); Ttime_list.append(time_val)
                
                if not Tdr_list:
                    print(f"\nNo valid volunteer data for analysis in {run_name_with_metric} at {intensity}. Skipping.")
                    continue

                Tdr = np.stack(Tdr_list, axis=2); Ttime = np.stack(Ttime_list, axis=2)
                binsM_indices = np.array([82, 84, 86, 88, 90, 92, 94, 96]) - 1
                TXD = np.mean(Tdr[binsM_indices, :, :], axis=(0, 2))
                noise_bins = np.setdiff1d(np.arange(shared_config['binsM_count']), np.union1d(binsM_indices, [0, 1]))
                FP = np.mean(Tdr[noise_bins, :, :], axis=(0, 2))

                plt.figure(figsize=(10, 6))
                plt.plot(TXD, '.k', markersize=10)
                plt.axhline(y=np.nanmean(TXD), color='r', linestyle=':', linewidth=2, label=f'Mean DR ({np.nanmean(TXD):.2f})')
                plt.ylabel('Detection Rate'); plt.xlabel('Parameter Set Index'); plt.title(f'Detection Rate Analysis ({run_name_with_metric} @ {intensity})')
                plt.legend(); plt.grid(True); plt.savefig(os.path.join(intensity_dir, f"{run_name_with_metric}_detection_rate.png")); plt.close()

                plt.figure(figsize=(10, 6))
                plt.plot(FP, '.k', markersize=10)
                plt.axhline(y=shared_config['FP_desejado'], color='r', linestyle=':', linewidth=2, label=f'Target FP ({shared_config["FP_desejado"]})')
                plt.ylabel('False Positive Rate'); plt.xlabel('Parameter Set Index'); plt.title(f'False Positive Rate Analysis ({run_name_with_metric} @ {intensity})')
                plt.legend(); plt.grid(True); plt.savefig(os.path.join(intensity_dir, f"{run_name_with_metric}_fpr.png")); plt.close()

                timeM_all = Ttime[binsM_indices, :, :]; timeM_all[timeM_all == -1] = shared_config['Mmax']
                timeM = np.mean(timeM_all, axis=(0, 2)); TXD_percent = TXD * 100
                valid_indices = ~np.isnan(TXD_percent) & ~np.isnan(timeM)
                
                df_all = pd.DataFrame(parametros[valid_indices], columns=['Mmin', 'Mstep', 'Mmax', 'NDC', 'alfa'])
                df_all['DR'] = TXD_percent[valid_indices]
                df_all['Time'] = timeM[valid_indices]
                df_all['FPR'] = FP[valid_indices] * 100
                all_run_dfs_for_intensity[run_name_with_metric] = df_all
        
        # --- STAGE 2: Standardize and generate ALL Pareto plots for the current intensity ---
        if not all_run_dfs_for_intensity:
            print(f"No data was generated for intensity {intensity}. Skipping plots.")
            sys.stdout.flush(); sys.stdout = original_stdout
            continue

        master_df = pd.concat(all_run_dfs_for_intensity.values())
        global_min_mmin = master_df['Mmin'].min()
        global_max_mmin = master_df['Mmin'].max()
        global_min_ndc = master_df['NDC'].min()
        global_max_ndc = master_df['NDC'].max()

        print(f"\n\n--- Generating Standardized Pareto Plots for {intensity} ---")
        all_pareto_fronts = {}

        # First loop: Generate individual plots
        for run_name, df_all in all_run_dfs_for_intensity.items():
            points_for_pareto = np.vstack([df_all['DR'], -df_all['Time']]).T
            _, idxs = paretoFront(points_for_pareto)
            
            df_pareto = df_all.iloc[idxs].sort_values(by='DR').reset_index(drop=True)
            if df_pareto.empty:
                print(f"No Pareto front found for {run_name} at {intensity}.")
                continue
            all_pareto_fronts[run_name] = df_pareto
            
            df_dominated = df_all.drop(index=idxs)

            fig, ax = plt.subplots(figsize=(18, 14))
            cmap_ndc = plt.cm.get_cmap('viridis_r')
            norm_ndc = mcolors.Normalize(vmin=global_min_ndc, vmax=global_max_ndc)

            def get_sizes(df, base_size, range_size):
                denom = global_max_mmin - global_min_mmin
                if denom < 1e-9: return np.full(len(df), base_size + range_size / 2)
                return base_size + range_size * (df['Mmin'] - global_min_mmin) / denom

            if not df_dominated.empty:
                ax.scatter(df_dominated['DR'], df_dominated['Time'], s=get_sizes(df_dominated, 40, 150), c=df_dominated['NDC'],
                           cmap=cmap_ndc, norm=norm_ndc, alpha=0.15, zorder=1, label='Dominated Solutions')
            
            ax.plot(df_pareto['DR'], df_pareto['Time'], '-', color='red', linewidth=2, alpha=0.6, zorder=2)
            
            df_high_fpr = df_pareto[df_pareto['FPR'] > 6.8181]
            df_low_fpr = df_pareto[df_pareto['FPR'] < 3.4090]
            df_mid_fpr = df_pareto[(df_pareto['FPR'] >= 3.4090) & (df_pareto['FPR'] <= 6.8181)]
            
            common_scatter_args = {'cmap': cmap_ndc, 'norm': norm_ndc, 'alpha': 0.85, 'edgecolors': 'black', 'linewidth': 0.7, 'zorder': 3}

            if not df_high_fpr.empty: ax.scatter(df_high_fpr['DR'], df_high_fpr['Time'], s=get_sizes(df_high_fpr, 150, 300), c=df_high_fpr['NDC'], marker='X', **common_scatter_args)
            if not df_low_fpr.empty: ax.scatter(df_low_fpr['DR'], df_low_fpr['Time'], s=get_sizes(df_low_fpr, 150, 300), c=df_low_fpr['NDC'], marker='^', **common_scatter_args)
            if not df_mid_fpr.empty: ax.scatter(df_mid_fpr['DR'], df_mid_fpr['Time'], s=get_sizes(df_mid_fpr, 150, 300), c=df_mid_fpr['NDC'], marker='o', **common_scatter_args)

            mappable = plt.cm.ScalarMappable(cmap=cmap_ndc, norm=norm_ndc)
            mappable.set_array([])
            
            cbar = fig.colorbar(mappable, ax=ax, orientation='horizontal', location='bottom', pad=0.1, aspect=40)
            cbar.set_label('NDC (Brighter/Yellow = Lower)', weight='bold', fontsize=12)

            ax.set_xlabel('Detection Rate (%)', fontsize=14)
            ax.set_ylabel('Mean Exam Time (Windows)', fontsize=14)
            ax.set_title(f'Pareto Front ({run_name} @ {intensity})', fontsize=18, weight='bold')
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # fig.tight_layout(rect=[0, 0, 1, 1])
            fig.savefig(os.path.join(intensity_dir, f"{run_name}_pareto_front.png"))
            plt.close(fig)
            print(f"Generated individual Pareto plot for {run_name}.")

        # Second Stage: Generate the combined "full" plot
        if not all_pareto_fronts:
            print(f"No Pareto fronts to combine for intensity {intensity}.")
        else:
            print(f"\n--- Plotting Full Pareto Comparison for {intensity} ---")
            fig, ax = plt.subplots(figsize=(22, 12))
            colors = plt.cm.get_cmap('tab20', len(all_pareto_fronts))
            cmap_ndc = plt.cm.get_cmap('viridis_r')
            norm_ndc = mcolors.Normalize(vmin=global_min_ndc, vmax=global_max_ndc)
            
            def get_sizes(df, base_size, range_size):
                denom = global_max_mmin - global_min_mmin
                if denom < 1e-9: return np.full(len(df), base_size + range_size / 2)
                return base_size + range_size * (df['Mmin'] - global_min_mmin) / denom
                
            for i, (run_name, df_pareto) in enumerate(all_pareto_fronts.items()):
                sizes = get_sizes(df_pareto, 60, 150)
                ax.plot(df_pareto['DR'], df_pareto['Time'], '-', color=colors(i), alpha=0.7, label=run_name, zorder=2)
                ax.scatter(df_pareto['DR'], df_pareto['Time'], s=sizes, c=df_pareto['NDC'],
                           cmap=cmap_ndc, norm=norm_ndc, edgecolors=colors(i), linewidth=1.5, zorder=3)
                
                for _, row in df_pareto.iterrows():
                    anno_text = (f"DR={row['DR']:.1f}%, FPR={row['FPR']:.1f}%\n"
                                 f"Exam={row['Time']:.1f}s, Mstep={int(row['Mstep'])}")
                    ax.annotate(anno_text, xy=(row['DR'], row['Time']), xytext=(0, -35),
                                textcoords='offset points', ha='center', va='top', fontsize=8,
                                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'))

            ax.set_title(f'Full Pareto Front Comparison ({intensity})', fontsize=20, weight='bold')
            ax.set_xlabel('Detection Rate (%)', fontsize=14)
            ax.set_ylabel('Mean Exam Time (Windows)', fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            mappable = plt.cm.ScalarMappable(cmap=cmap_ndc, norm=norm_ndc)
            mappable.set_array([])
            cbar = fig.colorbar(mappable, ax=ax, orientation='vertical', pad=0.02)
            cbar.set_label('NDC (Brighter/Yellow = Lower)', weight='bold', fontsize=12)
            
            handles, labels = ax.get_legend_handles_labels()
            legend1 = ax.legend(handles, labels, title='H0 Simulation Method & Metric', fontsize=12, 
                                bbox_to_anchor=(1.2, 0.9), loc='upper left')
            ax.add_artist(legend1)

            size_handles = [plt.scatter([],[], s=get_sizes(pd.DataFrame({'Mmin':[global_min_mmin]}), 60, 150)[0], c='gray', label=f'Smallest Mmin (~{global_min_mmin:.0f})'),
                            plt.scatter([],[], s=get_sizes(pd.DataFrame({'Mmin':[global_max_mmin]}), 60, 150)[0], c='gray', label=f'Largest Mmin (~{global_max_mmin:.0f})')]
            
            ax.legend(handles=size_handles, title='M_min (Larger Ball = Larger M_min)', 
                      fontsize=12, bbox_to_anchor=(1.2, 1.0), loc='upper left')

            # fig.tight_layout(rect=[0, 0, 0.85, 1])
            fig.savefig(os.path.join(intensity_dir, f"AAAA_full_pareto_comparison_{intensity}.png"))
            plt.close(fig)

        sys.stdout.flush()
        sys.stdout = original_stdout

    print(f"\n\nAll simulation runs are complete. Results are in '{master_output_dir}'.")


if __name__ == '__main__':
    if not os.path.isdir(PATH_EEG_DATA):
        print(f"Error: Data directory not found at '{PATH_EEG_DATA}'")
    else:
        main()