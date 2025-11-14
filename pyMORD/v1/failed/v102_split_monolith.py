import os
import numpy as np
import scipy.io as sio
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from numba import njit, prange
from tqdm import tqdm

# Global Configuration
DATA_PATH = r"C:\Users\alexa\experimental_data\todos\ENTRADAS_PATRICIA"
SIGNAL_FREQUENCIES = [81, 85, 89, 93, 83, 87, 91, 95]
ELECTRODES = ['Fz','F3','F4','F7','Fcz','Cz','C3','C4','Pz','P3','P4','T3','T4','T5','T6','Oz']
MAX_WINDOWS_PER_INTENSITY = {'70dB': 50, '60dB': 40, '50dB': 40, '40dB': 40, '30dB': 40, 'ESP': 40}

# Setup Plotting defaults
plt.rcParams['font.size'] = 12

@njit(fastmath=True)
def msc_fft(Y):
    """
    Optimized version of msc_fft. 
    Y: Input array (complex FFT data), shape (freq_bins, num_windows)
    """
    M = Y.shape[1]
    # Numerator: |sum over windows|^2
    num = np.abs(np.sum(Y, axis=1))**2
    # Denominator: M * sum(|Y|^2 over windows)
    den = M * np.sum(np.abs(Y)**2, axis=1)
    return num / den

@njit(fastmath=True)
def vc_msc_theoretical(M, alpha):
    """Theoretical critical value for MSC (assuming single channel MSC)."""
    return 1.0 - alpha**(1.0 / (M - 1.0))

@njit(fastmath=True)
def ets_kernel(ord_row, M_vals, critical_vals, NDC):
    """
    Core sequential test strategy kernel (replaces ETS.txt loop).
    ord_row: MSC values for one simulation run at window indices M_vals
    M_vals: Array of window indices (e.g., [20, 21, 22...])
    critical_vals: Array of corresponding critical values
    NDC: Number of consecutive detections needed
    """
    cont = 0
    # Iterate through the protocol steps
    for i in range(len(M_vals)):
        m_idx = M_vals[i] - 1 # Convert to 0-based index if M_vals are 1-based indices used in ord_row
        
        # If detection occurs at this step
        if ord_row[m_idx] > critical_vals[i]:
            cont += 1
        else:
            cont = 0
            
        if cont >= NDC:
            return 1.0, M_vals[i] # Detection, Time
            
    return 0.0, M_vals[-1] # No detection, return max time

@njit(parallel=True)
def run_ets_montecarlo(ord_matrix, M_vals, alpha, NDC):
    """
    Runs ETS in parallel over many Monte Carlo simulations.
    Replaces the loop in funcao_custo_v2 loop.
    """
    n_runs, _ = ord_matrix.shape
    dr_arr = np.zeros(n_runs)
    
    # Pre-calculate critical values for this alpha set
    crit_vals = np.empty(len(M_vals))
    for i in range(len(M_vals)):
        crit_vals[i] = vc_msc_theoretical(M_vals[i], alpha)
        
    for ii in prange(n_runs):
        # We use NDC directly in ets_kernel. ord_matrix indexing assumes full width Mmax
        dr, _ = ets_kernel(ord_matrix[ii], M_vals, crit_vals, NDC)
        dr_arr[ii] = dr
        
    return np.mean(dr_arr)

def generate_protocol_parameters(Mmax):
    """Replaces parametros_protocolo.txt"""
    P = []
    for M_step in range(1, Mmax):
        for Mmin in range(2, Mmax):
            k = (Mmax - Mmin) / M_step
            if k.is_integer() and k >= 0:
                P.append([Mmin, M_step, Mmax])
    
    # Add single test protocol
    P.append([Mmax, 1, Mmax])
    return np.array(P, dtype=int)

def cost_function_v2(alpha, ndc, mm_vals, ord_matrix, fp_target):
    """Replaces funcao_custo_v2.txt optimization target."""
    fp_calc = run_ets_montecarlo(ord_matrix, mm_vals, alpha[0], ndc)
    error = fp_calc - fp_target
    return 0.5 * error**2

def estimate_ndc(ndc_initial, alpha_test, fp_target, ord_matrix, mm_vals):
    """Replaces estimarNDC.txt."""
    n_runs = ord_matrix.shape[0]
    nnt_max = len(mm_vals)
    
    # Initial check
    fp = run_ets_montecarlo(ord_matrix, mm_vals, alpha_test, ndc_initial)
    if fp < fp_target and ndc_initial > 3:
        # Warning: large initial NDC reduced time
        ndc_initial = 1
        
    fp_history = {}
    ndc_found = None
    
    for ndc in range(int(ndc_initial), nnt_max + 1):
        fp_val = run_ets_montecarlo(ord_matrix, mm_vals, alpha_test, ndc)
        fp_history[ndc] = fp_val
        if fp_val < fp_target:
            ndc_found = ndc
            break
            
    if ndc_found is None:
        # Handle case where no NDC satisfies condition
        if len(fp_history) >= 2:
             # Simplistic fallback similar to interp1 in MATLAB code
             keys = sorted(fp_history.keys())
             vals = [fp_history[k] for k in keys]
             ndc_found = np.interp(fp_target, vals, keys)
        else:
            ndc_found = 1
    
    return ndc_found, fp_history.get(ndc_found, 0)

def compute_optimal_parameters(n_runs, Mmax, alpha_test, fp_target):
    """
    Main routine to pre-calculate optimal NDC and alpha for all protocols.
    """
    tj = 32 # Window size in points
    bin_freq = 8 # Target bin
    n_total = Mmax * tj
    
    print(f"Running Monte Carlo simulations with {n_runs} runs...")
    ord_matrix = np.zeros((n_runs, Mmax))
    
    # Simulate noise data under H0 (Null hypothesis)
    # Pre-generate all random data at once for efficiency
    x_noise = np.random.randn(n_runs, tj, Mmax)
    x_fft = np.fft.fft(x_noise, axis=1) # FFT along window time axis (axis 1)
    
    # We specifically look at the target bin `bin_freq` across the increasing windows M
    # Since msc_fft expects (freq, windows), we extract the relevant bin
    Y_bin = x_fft[:, bin_freq, :] # Shape (n_runs, Mmax)
    
    # Calculate MSC incrementally for M=2 to Mmax
    for i in range(n_runs):
        # We need cumulative MSC vector
        # Note: MSC implementation in MATLAB takes Y(bin, 1:M) and M.
        # We optimized this above to loop manually for clarity here
        for M in range(2, Mmax + 1):
            Y_slice = Y_bin[i, :M]
            # Apply formula inline for scalar slice
            num = np.abs(np.sum(Y_slice))**2
            den = M * np.sum(np.abs(Y_slice)**2)
            ord_matrix[i, M-1] = num / den
            
    # Generate parameter combinations
    P = generate_protocol_parameters(Mmax)
    
    # Filter or sort P just like in MATLAB to process efficiently
    # (Sorting skipped for brevity, but P is same structure)
    
    # Storage results
    results = {
        'P': P,
        'NDC_minimo': np.full(len(P), np.nan),
        'alfa_corrigido': np.full(len(P), np.nan),
        'cost': np.full(len(P), np.nan)
    }
    
    ndc_current = 1
    for i in tqdm(range(len(P)), desc="Optimizing Protocols"):
        Mmin, Mstep, Mmax_p = P[i]
        mm_vals = np.arange(Mmin, Mmax_p + 1, Mstep)
        
        if i > 0 and not np.isnan(results['NDC_minimo'][i-1]):
             ndc_current = max(int(results['NDC_minimo'][i-1]) - 5, 1)
             
        # Step 1: Find NDC
        ndc, fp = estimate_ndc(ndc_current, alpha_test, fp_target, ord_matrix, mm_vals)
        results['NDC_minimo'][i] = ndc
        
        # Step 2: Optimize Alpha
        res_opt = minimize(cost_function_v2, x0=[alpha_test], args=(ndc, mm_vals, ord_matrix, fp_target), 
                           method='CG', options={'maxiter': 50})
        
        results['alfa_corrigido'][i] = res_opt.x[0]
        results['cost'][i] = res_opt.fun
        
    return results, ord_matrix

@njit
def protocol_detection_fast(x_fft_bin, parameters):
    """
    Applies protocol detection for a single channel/freq bin.
    x_fft_bin: FFT data slice for bin indices
    parameters: array [Mmin, Mstep, Mmax, NDC, alpha]
    """
    num_protocols = parameters.shape[0]
    # `ord` calculation for the real signal up to its max length
    M_signal = x_fft_bin.shape[1]
    ord_vals = np.zeros(M_signal)
    for M in range(2, M_signal + 1):
        Y_slice = x_fft_bin[:, :M]
        # vectorized msc over frequency bins
        num = np.abs(np.sum(Y_slice, axis=1))**2
        den = M * np.sum(np.abs(Y_slice)**2, axis=1)
        ord_vals[M-1] = (num / den)[0] # Assuming single bin processed here for simplicity?
        # Wait, the original processed multiple bins at once.
        
    # Need clarification on x_fft_bin structure in real execution loops.
    # Let's revert to the higher level implementation to ensure correct shapes.
    pass 

def run_protocol_on_volunteers(volunteers, intensity, parameters):
    """
    Main loop over volunteers.
    """
    Mmax_proto = 240 # Defined in script
    fs_target = 256   # Usually 1s window = Fs points
    
    # Pre-allocate results storage
    # Assuming 100 bins being tracked
    n_vol = len(volunteers)
    n_proto = len(parameters)
    Tdr = np.zeros((100, n_proto, n_vol))
    Ttime = np.zeros((100, n_proto, n_vol))
    
    for i_vol, voluntario in enumerate(tqdm(volunteers, desc="Processing Vols")):
        # Construct file path
        file_name = os.path.join(DATA_PATH, f"{voluntario}{intensity}.mat")
        try:
            data = sio.loadmat(file_name)
        except FileNotFoundError:
            print(f"Missing: {file_name}")
            continue
            
        x = data['x']
        Fs = data['Fs'][0,0]
        freq_estim = data['freqEstim'][0] if 'freqEstim' in data else []
        binsM = data['binsM'][0] # 1-based indices in MATLAB
        
        # Select electrode (adjust index 0 for python)
        pos_ele = 0 
        x_ele = x[:, :, pos_ele]
        
        # Preprocessing: DC removal per window (window length = Fs)
        # MATLAB: x = x - repmat(mean(x), nfft, 1)
        nfft = Fs
        x_ele = x_ele - np.mean(x_ele, axis=0) # broadcasting works correctly here (windows on axis 1?)
        
        # MATLAB: x(:,1:2,:) = [] (remove first 2 seconds/windows)
        x_ele = x_ele[:, 2:]
        
        # Artifact removal (amplitude threshold)
        remoc = 0.1 / 200
        v_max = np.max(np.abs(x_ele), axis=0)
        clean_mask = v_max <= remoc
        x_ele = x_ele[:, clean_mask]
        x_ele = x_ele[:, :Mmax_proto]
        
        if x_ele.shape[1] < 5: # Skip if too few windows left
            continue
            
        # Run detection protocol
        xfft = np.fft.fft(x_ele, axis=0) # FFT along time (axis 0)
        
        # Calculate ORD for all bins up to bin 120
        bins_calc = 120
        n_windows = x_ele.shape[1]
        ord_vals = np.zeros((bins_calc, n_windows))
        for M in range(2, n_windows + 1):
            Y_current = xfft[:bins_calc, :M]
            ord_vals[:, M-1] = msc_fft(Y_current) # msc_fft takes (bins, windows)
            
        # Apply protocols
        for i_param in range(n_proto):
            p_row = parameters[i_param]
            Mmin, Mstep, Mmax_p = int(p_row[0]), int(p_row[1]), int(p_row[2])
            NDC = p_row[3]
            alpha = p_row[4] # Already corrected alpha
            
            mm_vals = np.arange(Mmin, Mmax_p + 1, Mstep, dtype=int)
            # Ensure mm_vals don't exceed actual data length
            mm_vals = mm_vals[mm_vals <= n_windows]
            
            if len(mm_vals) == 0:
                 Ttime[:, i_param, i_vol] = -1
                 continue
                 
            crit_vals = vc_msc_theoretical(mm_vals, alpha)
            
            # Loop over frequency bins
            for bin_idx in range(bins_calc):
                 dr_status, time_status = ets_kernel(ord_vals[bin_idx], mm_vals, crit_vals, NDC)
                 Tdr[bin_idx, i_param, i_vol] = dr_status
                 # If not detected, MATLAB returns Mmax (-1 in my mapping until calc)
                 Ttime[bin_idx, i_param, i_vol] = time_status if dr_status == 1.0 else -1
                 
    return Tdr, Ttime, binsM

def pareto_front(P):
    """
    Finds the pareto frontier of a set of points P (N x D).
    Assumes we want to minimize the objectives.
    For maximization, negate the input column before passing.
    """
    n_points, n_dim = P.shape
    is_dominated = np.zeros(n_points, dtype=bool)
    for i in range(n_points):
        for j in range(n_points):
            if i == j: continue
            if np.all(P[j] <= P[i]) and np.any(P[j] < P[i]):
                is_dominated[i] = True
                break
    
    idxs = np.where(~is_dominated)[0]
    return P[idxs], idxs

if __name__ == '__main__':
    # 1. Load or Compute Pre-calculated parameters
    Mmax_setup = 240
    alpha_setup = 0.05
    fp_target_setup = 0.05
    
    # Check if pre-calculated file exists
    params_file = f'NDC_AlfaCorrigido_Mmax{Mmax_setup}.npy'
    if os.path.exists(params_file):
        print("Loading pre-computed parameters...")
        # Load dictionary with 'P', 'NDC_minimo', 'alfa_corrigido' keys
        saved_res = np.load(params_file, allow_pickle=True).item()
        parameters = np.column_stack([
            saved_res['P'],
            saved_res['NDC_minimo'],
            saved_res['alfa_corrigido']
        ])
    else:
        print("Computing optimization parameters (takes time)...")
        res_params, _ = compute_optimal_parameters(1000, Mmax_setup, alpha_setup, fp_target_setup)
        # For demo, let's save it
        np.save(params_file, res_params)
        parameters = np.column_stack([
            res_params['P'],
            res_params['NDC_minimo'],
            res_params['alfa_corrigido']
        ])

    # Ensure alpha is set to 0.05 if no correction desired (MATLAB script option)
    # parameters[:, 4] = 0.05 

    # 2. Define volunteers and intensity
    volunteers = ['Ab', 'An', 'Bb', 'Er', 'Lu', 'So', 'Qu', 'Vi', 'Sa', 'Ti', 'Wr']
    intensity = '50dB'
    
    # 3. Run analysis
    Tdr, Ttime, binsM_matlab = run_protocol_on_volunteers(volunteers, intensity, parameters)
    
    # Convert binsM (1-based) to 0-based
    binsM_indices = (binsM_matlab - 1).astype(int)
    
    # Calculate TXD (Detection Rate at stimulation bins)
    # Tdr shape: (bins, n_proto, n_vol)
    # Average over volunteers (axis 2), then select stimulation bins
    txd_per_bin = np.mean(Tdr, axis=2) 
    TXD = np.mean(txd_per_bin[binsM_indices], axis=0) * 100
    
    # Calculate FP (False Positive Rate at non-stimulation bins)
    # Select range like MATLAB: binsR = binsM+1, but excluding initial bins
    binsR = np.arange(2, 100) # python 0-based (start=2 means index 2 -> bin 3)
    mask_R = np.isin(binsR, binsM_indices, invert=True)
    binsR_indices = binsR[mask_R]
    
    fp_per_bin = np.mean(Tdr, axis=2)
    FP = np.mean(fp_per_bin[binsR_indices], axis=0)
    
    # Calculate Mean Time for detection bins
    time_mat = Ttime[binsM_indices]
    # Replace -1 with Mmax (240) for mean calculation
    time_mat[time_mat == -1] = Mmax_setup
    timeM = np.mean(np.mean(time_mat, axis=2), axis=0) # Mean over vol, mean over bins
    
    # 4. Plotting Pareto Front
    # Minimize (-TXD, timeM) -> maximize TXD, minimize time
    objectives = np.column_stack([-TXD, timeM])
    pareto_pts, pareto_idxs = pareto_front(objectives)
    
    # Sort by detection rate for clean line plotting
    sort_idx = np.argsort(-pareto_pts[:, 0]) # sort descending TXD
    pareto_pts = pareto_pts[sort_idx]
    pareto_idxs = pareto_idxs[sort_idx]

    plt.figure(figsize=(10, 6))
    plt.scatter(TXD, timeM, c='k', s=10, label='Protocols')
    plt.plot(TXD[pareto_idxs], timeM[pareto_idxs], '-or', linewidth=2, label='Pareto Front')
    plt.plot([0, 100], [Mmax_setup, Mmax_setup], '--k', label='Max Time')
    plt.xlabel('Detection Rate (%)')
    plt.ylabel('Mean Exam Time (s)')
    plt.title(f'Protocol Optimization (Intensity: {intensity})')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Print optimal points
    print("\nOptimal Pareto Points:")
    for idx in pareto_idxs:
        p_row = parameters[idx]
        print(f"Protocol[{int(p_row[0])}, {int(p_row[1])}]: DR={TXD[idx]:.2f}%, Time={timeM[idx]:.2f}s")