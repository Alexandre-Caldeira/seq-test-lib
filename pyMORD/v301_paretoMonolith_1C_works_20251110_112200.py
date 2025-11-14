# =============================================================================
# Section 1: Imports and Configuration
# =============================================================================
import os
import numpy as np
import matplotlib.pyplot as plt
import mat73
from scipy.stats import chi2, f as finv
import itertools
from tqdm import tqdm

# --- User-editable paths ---
path_mat_file = r"C:\PPGEE\Assessing CGST on ASSR\Numero_Deteccoes_consecutiva_H"
path_eeg_data = r"C:\Users\alexa\experimental_data\todos\ENTRADAS_PATRICIA"

# =============================================================================
# Section 2: Ported Metrics and Function Definitions
# =============================================================================

# --- Metrics Ported from Code A ---
def msweep(matrix, r):
    A = np.copy(matrix).astype(np.complex128)
    for k in range(r):
        d = A[k, k]
        if np.abs(d) < 1e-12: d = 1e-12
        col_k, row_k = A[:, k].copy(), A[k, :].copy()
        A -= np.outer(col_k, row_k) / d
        A[k, :], A[:, k], A[k, k] = row_k / d, -col_k / d, 1.0 / d
    return A

def _calculate_univariate_msc(data, tj, fs, alpha):
    M = data.shape[0] // tj
    if M <= 1: return np.zeros(tj // 2 + 1), 1.0
    epochs = data.flatten()[:M*tj].reshape(M, tj)
    nf = tj // 2 + 1
    Y = np.fft.fft(epochs, axis=1)[:, :nf]
    sum_Y, sum_Y_sq_mag = np.sum(Y, axis=0), np.sum(np.abs(Y)**2, axis=0)
    msc_values = np.zeros(nf)
    valid_idx = sum_Y_sq_mag > 1e-12
    msc_values[valid_idx] = np.abs(sum_Y[valid_idx])**2 / (M * sum_Y_sq_mag[valid_idx])
    return msc_values, (1 - alpha**(1/(M-1)) if M > 1 else 1.0)

def _calculate_univariate_csm(data, tj, fs, alpha):
    M = data.shape[0] // tj
    if M <= 1: return np.zeros(tj // 2 + 1), 1.0
    epochs = data.flatten()[:M * tj].reshape(M, tj)
    Y = np.fft.fft(epochs, axis=1)[:, :tj // 2 + 1]
    teta = np.angle(Y)
    sum_cos, sum_sin = np.sum(np.cos(teta), axis=0), np.sum(np.sin(teta), axis=0)
    csm_values = (sum_cos**2 + sum_sin**2) / (M**2)
    return csm_values, (chi2.ppf(1 - alpha, 2) / (2 * M) if M > 0 else 1.0)

def calculate_mmsc(data, tj, fs, alpha=0.05):
    n_samples, n_channels = data.shape if data.ndim > 1 else (data.shape[0], 1)
    if n_channels == 1: return _calculate_univariate_msc(data, tj, fs, alpha)
    M, nf = n_samples // tj, tj // 2 + 1
    if M <= n_channels: return np.zeros(nf), 1.0
    epochs = data[:M*tj, :].T.reshape(n_channels, M, tj)
    Sfft = np.fft.fft(epochs, axis=2)[:, :, :nf]
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
    Fcrit = finv.ppf(1-alpha, 2*n_channels, 2*(M-n_channels))
    return mmsc, Fcrit / (((M-n_channels)/n_channels) + Fcrit)

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

# --- Original and Helper Functions ---
def dipolos_py(x):
    n_samples, n_channels = x.shape
    n_combs = n_channels * (n_channels - 1) // 2
    y = np.zeros((n_samples, n_channels + n_combs))
    y[:, :n_channels] = x
    idx = n_channels
    for i, j in itertools.combinations(range(n_channels), 2):
        y[:, idx] = x[:, i] - x[:, j]
        idx += 1
    return y

def VC_MSC_py(M, alfa):
    M_safe = np.asarray(M, dtype=float)
    M_safe[M_safe <= 1] = np.nan
    return 1 - np.power(alfa, 1. / (M_safe - 1))

def ETS_py(ord_values, MM, alfa, NDC, vc_values):
    NDC = np.ceil(NDC)
    det = ord_values > vc_values
    consecutive_detections = 0
    for i, is_detected in enumerate(det):
        consecutive_detections = consecutive_detections + 1 if is_detected else 0
        if consecutive_detections >= NDC:
            return 1, MM[i]
    return 0, MM[-1]

def protocolo_deteccao_py(x, fs, parametros, metric_name):
    num_bins_protocol, num_windows_total = 120, x.shape[1]
    time_pts = x.shape[0]
    
    METRIC_FUNCTIONS = {
        'MMSC': calculate_mmsc, 'MCSM': calculate_mcsm,
        'D-aMSC': calculate_d_amsc, 'D-aCSM': calculate_d_acsm
    }
    metric_func = METRIC_FUNCTIONS[metric_name]
    
    ord_matrix = np.zeros((num_bins_protocol, num_windows_total - 1))
    crit_matrix = np.zeros((num_bins_protocol, num_windows_total - 1))

    for M in range(2, num_windows_total + 1):
        if x.ndim == 3: # Dipole data
            data_cont = x[:, :M, :].transpose(1, 0, 2).reshape(-1, x.shape[2])
        else: # Channel data
            data_cont = x[:, :M].T.reshape(-1, 1) if x.ndim == 2 else x[:, :M].T.reshape(-1, x.shape[2])
        
        # Calculate metric for this M
        if data_cont.shape[0] >= time_pts:
            ord_vals, crit_val = metric_func(data_cont, time_pts, fs, alpha=0.05)
            ord_matrix[:, M - 2] = ord_vals[:num_bins_protocol]

    num_param_sets = parametros.shape[0]
    dr = np.zeros((num_bins_protocol, num_param_sets))
    time = np.zeros((num_bins_protocol, num_param_sets))

    for ii in range(num_param_sets):
        Mmin, Mstep, Mmax, NDC, alfa_corr = parametros[ii, :]
        MM = np.arange(Mmin, Mmax + 1, Mstep, dtype=int)
        if MM.size == 0 or np.any(MM > num_windows_total): continue
        
        vc = VC_MSC_py(MM, alfa_corr)
        for ll in range(num_bins_protocol):
            dr[ll, ii], time[ll, ii] = ETS_py(ord_matrix[ll, MM - 2], MM, alfa_corr, NDC, vc)
            
    return dr, time

def pareto_front_py(points):
    is_pareto = np.ones(points.shape[0], dtype=bool)
    for i, c in enumerate(points):
        if is_pareto[i]:
            is_pareto[is_pareto] = np.any(points[is_pareto] > c, axis=1) | np.all(points[is_pareto] == c, axis=1)
            is_pareto[i] = True
    return points[is_pareto], np.where(is_pareto)[0]


# =============================================================================
# Section 3: Main Script
# =============================================================================

# --- Script Parameters ---
Vvoluntario = ['Ab', 'An', 'Er', 'Qu', 'Sa', 'Ti', 'Wr']
Intensidade = ['50dB']
Mmax = 240
alfa, FP_desejado = 0.05, 0.05
pos_ele = 1
ganho, remoc = 200, 0.1 / 200
metrics_to_run = ['MMSC', 'MCSM', 'D-aMSC', 'D-aCSM']
target_electrodes = ['Fz', 'Cz', 'Pz', 'Oz', 'T3', 'T4']
ELECTRODE_LIST = ['Fz','F3','F4','F7','Fcz','Cz','C3','C4','Pz','P3','P4','T3','T4','T5','T6','Oz']

# --- Load Protocol Parameters ---
mat_filename = f"NDC_AlfaCorrigido_Mmax{Mmax}_alfa_{alfa}_FPdesejado{FP_desejado}.mat"
mat_filepath = os.path.join(path_mat_file, mat_filename)
try:
    mat_contents = mat73.loadmat(mat_filepath)
    P, alfa_corrigido, NDC_minimo = mat_contents['P'], mat_contents['alfa_corrigido'], mat_contents['NDC_minimo']
except Exception as e:
    raise FileNotFoundError(f"Could not load or parse parameter file: {mat_filepath}\nError: {e}")

parametros = np.hstack((P, np.array(NDC_minimo).reshape(-1, 1), np.array(alfa_corrigido).reshape(-1, 1)))
electrode_indices = [ELECTRODE_LIST.index(e) for e in target_electrodes]
results = {}

# --- Main Loop over Metrics and Volunteers ---
for metric in metrics_to_run:
    print(f"\n===== Running analysis for metric: {metric} =====")
    Tdr = np.zeros((120, parametros.shape[0], len(Vvoluntario)))
    Ttime = np.zeros((120, parametros.shape[0], len(Vvoluntario)))

    for cont_vol, voluntario_code in enumerate(tqdm(Vvoluntario, desc=f"Volunteers ({metric})")):
        eeg_filepath = os.path.join(path_eeg_data, f"{voluntario_code}{Intensidade[0]}.mat")
        eeg_data = mat73.loadmat(eeg_filepath)
        x_all_ele, Fs, binsM_analysis = eeg_data['x'], float(eeg_data['Fs']), (np.array(eeg_data['binsM']).flatten() - 1).astype(int)
        
        x_subset = x_all_ele[:, :, electrode_indices]
        
        if metric in ['D-aMSC', 'D-aCSM']:
            n_win = x_subset.shape[1]
            input_data = np.array([dipolos_py(x_subset[:, i, :]) for i in range(n_win)]).transpose(1,0,2)
        else:
             input_data = x_subset
        
        # Preprocessing
        input_data -= input_data.mean(axis=0, keepdims=True)
        clean_indices = np.max(np.abs(input_data), axis=0) <= remoc
        clean_data = input_data[:, clean_indices.all(axis=1) if input_data.ndim==3 else clean_indices][:, :Mmax]
        if clean_data.shape[1] < 2: continue
        
        dr, time = protocolo_deteccao_py(clean_data, Fs, parametros, metric)
        Tdr[:, :, cont_vol], Ttime[:, :, cont_vol] = dr, time
    
    # --- Performance Calculation ---
    TXD = np.mean(Tdr[binsM_analysis, ...], axis=(0, 2))
    all_bins = np.arange(100)
    binsR = np.setdiff1d(all_bins, binsM_analysis)
    binsR = binsR[binsR > 1]
    FP = np.mean(Tdr[binsR, ...], axis=(0, 2))

    timeM = Ttime[binsM_analysis, :, :]
    timeM[timeM == -1] = Mmax
    timeM_mean = np.mean(timeM, axis=(0, 2))
    
    results[metric] = {'txd': TXD * 100, 'fpr': FP * 100, 'time': timeM_mean}

# =============================================================================
# Section 4: Analysis and Plotting
# =============================================================================
colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

# Plot 1: TPR vs. Time (Pareto)
fig1, ax1 = plt.subplots(figsize=(14, 9))
for i, (metric, data) in enumerate(results.items()):
    points = np.vstack((data['txd'], -data['time'])).T
    p_front, _ = pareto_front_py(points)
    p_front = p_front[np.argsort(p_front[:, 0])]
    ax1.plot(p_front[:, 0], -p_front[:, 1], 'o-', color=colors[i], label=f'{metric} Pareto Front')

ax1.set_title(f'Pareto Front Comparison: Detection Rate vs. Exam Time ({Intensidade[0]})', fontsize=16)
ax1.set_xlabel('True Positive Rate (TPR / Detection Rate, %)', fontsize=12)
ax1.set_ylabel('Mean Exam Time (s)', fontsize=12)
ax1.grid(True, linestyle='--'); ax1.legend(); ax1.set_ylim(bottom=0); ax1.set_xlim(left=0)
plt.tight_layout(); plt.show()

# Plot 2: FPR vs. Time
fig2, ax2 = plt.subplots(figsize=(14, 9))
for i, (metric, data) in enumerate(results.items()):
    sorted_indices = np.argsort(data['time'])
    ax2.plot(data['time'][sorted_indices], data['fpr'][sorted_indices], 'o-', color=colors[i], label=metric, alpha=0.8)

ax2.set_title(f'Performance Comparison: False Positive Rate vs. Exam Time ({Intensidade[0]})', fontsize=16)
ax2.set_xlabel('Mean Exam Time (s)', fontsize=12)
ax2.set_ylabel('False Positive Rate (FPR, %)', fontsize=12)
ax2.grid(True, linestyle='--'); ax2.legend(); ax2.set_ylim(bottom=0); ax2.set_xlim(left=0)
plt.tight_layout(); plt.show()