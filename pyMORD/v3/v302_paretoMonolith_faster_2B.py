import os
import numpy as np
import matplotlib.pyplot as plt
import mat73
from scipy.stats import chi2, f as finv
import itertools
from tqdm import tqdm
import multiprocessing

# OPTIMIZED: Import Numba for JIT compilation
from numba import njit, float64, int64, complex128, types

# --- User-editable paths ---
path_mat_file = r"C:\PPGEE\Assessing CGST on ASSR\Numero_Deteccoes_consecutiva_H"
path_eeg_data = r"C:\Users\alexa\experimental_data\todos\ENTRADAS_PATRICIA"

# =============================================================================
# Section 2: JIT-Compiled Metrics and Function Definitions
# =============================================================================

# --- Metrics and Helpers with Numba Compilation ---
@njit(complex128[:,:](complex128[:,:], int64), cache=True)
def msweep_njit(matrix, r):
    A = matrix.copy()
    for k in range(r):
        d = A[k, k]
        if np.abs(d) < 1e-12: d = 1e-12
        col_k, row_k = A[:, k].copy(), A[k, :].copy()
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                A[i, j] -= col_k[i] * row_k[j] / d
        A[k, :], A[:, k], A[k, k] = row_k / d, -col_k / d, 1.0 / d
    return A

# --- Core Metric Functions (Ported from Code A/B) ---
def _calculate_univariate_msc(data, tj, fs, alpha):
    M = data.shape[0] // tj
    if M <= 1: return np.zeros(tj // 2 + 1), 1.0
    epochs = data.flatten()[:M*tj].reshape(M, tj); nf = tj // 2 + 1
    Y = np.fft.fft(epochs, axis=1)[:, :nf]
    sum_Y, sum_Y_sq_mag = np.sum(Y, axis=0), np.sum(np.abs(Y)**2, axis=0)
    msc = np.zeros(nf); valid_idx = sum_Y_sq_mag > 1e-12
    msc[valid_idx] = np.abs(sum_Y[valid_idx])**2 / (M * sum_Y_sq_mag[valid_idx])
    return msc, (1 - alpha**(1/(M-1)) if M > 1 else 1.0)

def _calculate_univariate_csm(data, tj, fs, alpha):
    M = data.shape[0] // tj
    if M <= 1: return np.zeros(tj // 2 + 1), 1.0
    epochs = data.flatten()[:M * tj].reshape(M, tj)
    Y = np.fft.fft(epochs, axis=1)[:, :tj // 2 + 1]
    teta = np.angle(Y)
    sum_cos, sum_sin = np.sum(np.cos(teta), axis=0), np.sum(np.sin(teta), axis=0)
    return (sum_cos**2 + sum_sin**2)/(M**2), (chi2.ppf(1-alpha, 2)/(2*M) if M > 0 else 1.0)

def calculate_mmsc(data, tj, fs, alpha=0.05):
    n_samples, n_channels = data.shape if data.ndim > 1 else (data.shape[0], 1)
    if n_channels == 1: return _calculate_univariate_msc(data, tj, fs, alpha)
    M, nf = n_samples // tj, tj // 2 + 1
    if M <= n_channels: return np.zeros(nf), 1.0
    epochs = data[:M*tj, :].T.reshape(n_channels, M, tj)
    Sfft = np.fft.fft(epochs, axis=2)[:, :, :nf]; mmsc = np.zeros(nf)
    for kf in range(nf):
        Sfft_slice = Sfft[:, :, kf]
        spec_a = np.zeros((n_channels + 1, n_channels + 1), dtype=np.complex128)
        spec_a[:n_channels, :n_channels] = Sfft_slice @ Sfft_slice.conj().T
        V = np.sum(Sfft_slice, axis=1)
        spec_a[n_channels, :n_channels], spec_a[:n_channels, n_channels] = V.conj(), V
        spec_a[n_channels, n_channels] = 1
        spec_as = msweep_njit(spec_a, n_channels) # Call JIT version
        mmsc[kf] = (1 - np.real(spec_as[n_channels, n_channels])) / M
    Fcrit = finv.ppf(1-alpha, 2*n_channels, 2*(M-n_channels))
    return mmsc, Fcrit / (((M-n_channels)/n_channels) + Fcrit)

def calculate_mcsm(data, tj, fs, alpha=0.05):
    n_samples, n_channels = data.shape if data.ndim > 1 else (data.shape[0], 1)
    if n_channels == 1: return _calculate_univariate_csm(data, tj, fs, alpha)
    M, nf = n_samples // tj, tj // 2 + 1
    if M == 0: return np.zeros(nf), 1.0
    Y = np.fft.fft(data[:M*tj, :].reshape(M, tj, n_channels), axis=1)[:, :nf, :]
    teta = np.angle(Y)
    C_mean = np.mean(np.cos(teta), axis=2); S_mean = np.mean(np.sin(teta), axis=2)
    teta_med = np.arctan2(S_mean, C_mean)
    sum_cos, sum_sin = np.sum(np.cos(teta_med), axis=0), np.sum(np.sin(teta_med), axis=0)
    return (sum_cos**2+sum_sin**2)/(M**2), chi2.ppf(1-alpha, 2*n_channels)/(2*M*n_channels)

def calculate_d_amsc(data, tj, fs, alpha=0.05):
    if data.ndim == 1 or data.shape[1] < 2: return _calculate_univariate_msc(data, tj, fs, alpha)
    mscs = [_calculate_univariate_msc(data[:, i], tj, fs, alpha)[0] for i in range(data.shape[1])]
    M = data.shape[0] // tj
    return np.mean(np.array(mscs), axis=0), (1-alpha**(1/(M-1)) if M > 1 else 1.0)

def calculate_d_acsm(data, tj, fs, alpha=0.05):
    if data.ndim == 1 or data.shape[1] < 2: return _calculate_univariate_csm(data, tj, fs, alpha)
    csms = [_calculate_univariate_csm(data[:, i], tj, fs, alpha)[0] for i in range(data.shape[1])]
    M = data.shape[0] // tj
    return np.mean(np.array(csms), axis=0), (chi2.ppf(1-alpha, 2)/(2*M) if M > 0 else 1.0)

# --- Other JIT-Compiled Helper Functions ---
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

@njit(float64[:](float64[:], float64), cache=True)
def VC_MSC_njit(M, alfa):
    return 1 - np.power(alfa, 1. / (M - 1))

@njit(types.UniTuple(int64, 2)(float64[:], int64[:], float64, float64, float64[:]), cache=True)
def ETS_njit(ord_values, MM, alfa, NDC, vc_values):
    NDC = np.ceil(NDC)
    det = ord_values > vc_values
    consecutive_detections = 0
    for i, is_detected in enumerate(det):
        consecutive_detections = consecutive_detections + 1 if is_detected else 0
        if consecutive_detections >= NDC:
            return 1, MM[i]
    return 0, MM[-1]

def protocolo_deteccao_py(x, fs, parametros, metric_name):
    num_bins, num_windows = 120, x.shape[1]
    time_pts = x.shape[0]
    
    METRIC_FUNCTIONS = {
        'MMSC': calculate_mmsc, 'MCSM': calculate_mcsm, 
        'D-aMSC': calculate_d_amsc, 'D-aCSM': calculate_d_acsm
    }
    metric_func = METRIC_FUNCTIONS[metric_name]
    
    ord_matrix = np.zeros((num_bins, num_windows - 1))
    for M in range(2, num_windows + 1):
        if x.ndim == 3: # Dipole data
            data_cont = x[:,:M,:].transpose(1,0,2).reshape(-1, x.shape[2])
        else: # Channel data, ensuring it's 2D for the metric function
            data_cont = x[:,:M].T.reshape(-1, x.shape[-1])

        if data_cont.shape[0] >= time_pts:
            ord_vals, _ = metric_func(data_cont, time_pts, fs, alpha=0.05)
            ord_matrix[:, M - 2] = ord_vals[:num_bins]

    dr, time = np.zeros((num_bins, len(parametros))), np.zeros((num_bins, len(parametros)))
    for ii in range(len(parametros)):
        Mmin, Mstep, Mmax, NDC, alfa_corr = parametros[ii, :]
        MM = np.arange(Mmin, Mmax + 1, Mstep, dtype=int)
        if MM.size == 0 or np.any(MM-2 >= ord_matrix.shape[1]): continue
        vc = VC_MSC_njit(MM.astype(np.float64), alfa_corr)
        for ll in range(num_bins):
            dr[ll, ii], time[ll, ii] = ETS_njit(ord_matrix[ll, MM - 2], MM, alfa_corr, NDC, vc)
    return dr, time

@njit(types.Tuple((float64[:,:], int64[:]))(float64[:,:]), cache=True)
def pareto_front_njit(points):
    n_points = points.shape[0]
    is_pareto = np.ones(n_points, dtype=types.boolean)
    for i in range(n_points):
        if is_pareto[i]:
            # A point is not dominated if there is no other point that is better in all objectives.
            # So, we find points that are NOT dominated by point i.
            # A point j dominates i if all(j >= i) and any(j > i)
            is_pareto[is_pareto] = np.any(points[is_pareto] > points[i], axis=1) | np.all(points[is_pareto] == points[i], axis=1)
            is_pareto[i] = True # A point does not dominate itself
    pareto_indices = np.where(is_pareto)[0]
    return points[pareto_indices], pareto_indices

# =============================================================================
# Section 3: Main Script (Parallelized)
# =============================================================================

def process_volunteer_c(args):
    """Encapsulates the logic for processing a single volunteer for parallel execution."""
    voluntario_code, Intensidade, Mmax, remoc, electrode_indices, parametros, metric = args
    try:
        eeg_filepath = os.path.join(path_eeg_data, f"{voluntario_code}{Intensidade}.mat")
        eeg_data = mat73.loadmat(eeg_filepath)
        x_all_ele, Fs, binsM = eeg_data['x'], float(eeg_data['Fs']), (np.array(eeg_data['binsM']).flatten() - 1).astype(int)
        
        x_subset = x_all_ele[:, :, electrode_indices]
        
        if metric in ['D-aMSC', 'D-aCSM']:
            input_data = np.array([dipolos_njit(x_subset[:, i, :]) for i in range(x_subset.shape[1])]).transpose(1,0,2)
        else:
            input_data = x_subset
        
        input_data -= input_data.mean(axis=0, keepdims=True)
        # Handle 2D or 3D artifact rejection
        axis_to_check = (0, 2) if input_data.ndim == 3 else 0
        clean_indices = np.max(np.abs(input_data), axis=axis_to_check) <= remoc
        clean_data = input_data[:, clean_indices][:, :Mmax]
        
        if clean_data.shape[1] < 2: return None
            
        dr, time = protocolo_deteccao_py(clean_data, Fs, parametros, metric)
        return dr, time, binsM
    except Exception as e:
        print(f"Warning: Failed to process {voluntario_code} for metric {metric}. Error: {e}")
        return None

def main():
    Vvoluntario = ['Ab', 'An', 'Er', 'Qu', 'Sa', 'Ti', 'Wr']
    Intensidade, Mmax, alfa, FP_desejado, remoc = ['50dB'], 240, 0.05, 0.05, 0.1 / 200
    metrics_to_run = ['MMSC', 'MCSM', 'D-aMSC', 'D-aCSM']
    target_electrodes = ['Fz', 'Cz', 'Pz', 'Oz', 'T3', 'T4']
    ELECTRODE_LIST = ['Fz','F3','F4','F7','Fcz','Cz','C3','C4','Pz','P3','P4','T3','T4','T5','T6','Oz']
    
    mat_contents = mat73.loadmat(os.path.join(path_mat_file, f"NDC_AlfaCorrigido_Mmax{Mmax}_alfa_{alfa}_FPdesejado{FP_desejado}.mat"))
    parametros = np.hstack((mat_contents['P'], np.array(mat_contents['NDC_minimo']).reshape(-1, 1), np.array(mat_contents['alfa_corrigido']).reshape(-1, 1)))
    electrode_indices = [ELECTRODE_LIST.index(e) for e in target_electrodes]
    results = {}

    for metric in metrics_to_run:
        print(f"\n===== Running analysis for metric: {metric} =====")
        
        tasks = [(vol, Intensidade[0], Mmax, remoc, electrode_indices, parametros, metric) for vol in Vvoluntario]
        Tdr_all, Ttime_all, binsM = [], [], None
        
        with multiprocessing.Pool() as pool:
            pool_results = list(tqdm(pool.imap(process_volunteer_c, tasks), total=len(tasks), desc=f"Volunteers ({metric})"))

        for res in pool_results:
            if res:
                dr, time, bM = res
                Tdr_all.append(dr); Ttime_all.append(time)
                if binsM is None: binsM = bM
        
        if not Tdr_all: 
            print(f"Warning: No valid data for metric {metric} after processing all volunteers.")
            continue

        Tdr, Ttime = np.stack(Tdr_all, axis=-1), np.stack(Ttime_all, axis=-1)
        
        TXD = np.mean(Tdr[binsM, ...], axis=(0, 2)) * 100
        binsR = np.setdiff1d(np.arange(100), binsM); binsR = binsR[binsR > 1]
        FP = np.mean(Tdr[binsR, ...], axis=(0, 2)) * 100
        timeM = Ttime[binsM, :, :]; timeM[timeM == -1] = Mmax
        timeM_mean = np.mean(timeM, axis=(0, 2))
        results[metric] = {'txd': TXD, 'fpr': FP, 'time': timeM_mean}
    
    # --- Plotting Section ---
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    # Plot 1: TPR vs. Time (Pareto Front)
    fig1, ax1 = plt.subplots(figsize=(14, 9))
    for i, (metric, data) in enumerate(results.items()):
        points = np.vstack((data['txd'], -data['time'])).T
        p_front, _ = pareto_front_njit(points)
        p_front = p_front[np.argsort(p_front[:, 0])]
        ax1.plot(p_front[:, 0], -p_front[:, 1], 'o-', color=colors[i], label=f'{metric} Pareto Front')
    
    ax1.set_title(f'Pareto Front: Detection Rate vs. Exam Time ({Intensidade[0]})', fontsize=16)
    ax1.set_xlabel('True Positive Rate (TPR, %)', fontsize=12)
    ax1.set_ylabel('Mean Exam Time (s)', fontsize=12)
    ax1.grid(True, linestyle='--'); ax1.legend(); ax1.set_ylim(bottom=0); ax1.set_xlim(left=0)
    plt.tight_layout()
    plt.show()

    # Plot 2: FPR vs. Time
    fig2, ax2 = plt.subplots(figsize=(14, 9))
    for i, (metric, data) in enumerate(results.items()):
        sorted_indices = np.argsort(data['time'])
        ax2.plot(data['time'][sorted_indices], data['fpr'][sorted_indices], 'o-', color=colors[i], label=metric, alpha=0.8)

    ax2.set_title(f'Performance: False Positive Rate vs. Exam Time ({Intensidade[0]})', fontsize=16)
    ax2.set_xlabel('Mean Exam Time (s)', fontsize=12)
    ax2.set_ylabel('False Positive Rate (FPR, %)', fontsize=12)
    ax2.grid(True, linestyle='--'); ax2.legend(); ax2.set_ylim(bottom=0); ax2.set_xlim(left=0)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    multiprocessing.freeze_support() # Necessary for Windows/macOS compatibility
    main()