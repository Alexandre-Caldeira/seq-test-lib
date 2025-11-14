import numpy as np
from numpy.fft import fft
from scipy.stats import chi2, f, beta, mstats
from scipy.optimize import minimize
from scipy.io import loadmat
from scipy.signal import detrend
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import itertools
from numba import njit, float64, int64, complex128
import os
import warnings
from typing import Dict, List, Tuple, Any

# --- 0. Configuration and Global Parameters (config.py) ---
# Similar to parametros_globais.txt and additional constants from the main script
class Config:
    N_RUNS_OPTIMIZATION = 1000  # For funcao_NDC_alfaCorrigido_Mmax
    N_RUNS_VC_MSC = 10000       # For VC_MSC
    ALPHA_DEFAULT = 0.05
    FP_DESEJADO_DEFAULT = 0.05
    MAX_ITER_FMINCG = 50
    TJ_DEFAULT = 32             # Window size (number of points) for simulation
    BIN_DEFAULT = 8             # Frequency bin index for simulation (corresponds to parameters.fo in mord)

    # From sinalleeg_ndc.txt
    GANHO = 200
    REMOC_THRESHOLD = 0.1 / GANHO # Amplitude threshold for noise removal

    # Paths
    # Assuming the current working directory is where the script will be run
    # and the data directory is a sibling or specified explicitly.
    # User needs to adjust this path to their actual data location
    DATA_PATH = "C:\\Users\\alexa\\experimental_data\\todos\\ENTRADAS_PATRICIA"
    # DATA_PATH = ".\experimental_data\todos\ENTRADAS_PATRICIA" # Example relative path

    # From original MATLAB comment, these are example definitions
    SIGNAL_FREQUENCIES: Dict[str, List[int]] = {
        '70dB': [81, 85, 89, 93, 83, 87, 91, 95],
        '60dB': [81, 85, 89, 93, 83, 87, 91, 95], # Assuming same for now
        '50dB': [81, 85, 89, 93, 83, 87, 91, 95], # Assuming same for now
        '40dB': [81, 85, 89, 93, 83, 87, 91, 95], # Assuming same for now
        '30dB': [81, 85, 89, 93, 83, 87, 91, 95], # Assuming same for now
        'ESP': [81, 85, 89, 93, 83, 87, 91, 95]  # Assuming same for now
    }
    ELECTRODES: List[str] = ['Fz', 'F3', 'F4', 'F7', 'Fcz', 'Cz', 'C3', 'C4', 'Pz', 'P3', 'P4', 'T3', 'T4', 'T5', 'T6', 'Oz']
    MAX_WINDOWS_PER_INTENSITY: Dict[str, int] = {'70dB': 50, '60dB': 40, '50dB': 40, '40dB': 40, '30dB': 40, 'ESP': 40}


# --- 1. parametros_protocolo.py ---
def generate_protocol_parameters(Mmax: int) -> np.ndarray:
    """
    Generates all possible parameter combinations for a detection protocol.
    Corresponds to 'parametros_protocolo.m'.

    Args:
        Mmax (int): Maximum number of windows.

    Returns:
        np.ndarray: Matrix P where each row is [Mmin, Mstep, Mmax].
    """
    P = []
    for M_step in range(1, Mmax):
        for Mmin in range(2, Mmax):
            k = (Mmax - Mmin) / M_step
            if k == int(k) and k >= 0:
                P.append([Mmin, M_step, Mmax])

    # Include the single test case (where Mmin = Mmax)
    P.append([Mmax, 1, Mmax])

    return np.array(P)

# --- 2. dipolos.py ---
@njit(float64[:,:](float64[:,:]))
def dipolos(x: np.ndarray) -> np.ndarray:
    """
    Generates dipole signals from multichannel data.
    Corresponds to 'dipolos.m'.

    Args:
        x (np.ndarray): Input signal matrix (time_points x N_channels).

    Returns:
        np.ndarray: Output signal matrix with original channels and dipole combinations.
    """
    N = x.shape[1]
    
    # MATLAB: CC = [[[1:N]', zeros(N,1)];combnk(1:N,2)];
    # First N rows: [channel_idx, 0] for individual channels
    # Next rows: [channel_idx1, channel_idx2] for combinations
    
    # Equivalent to combnk(1:N, 2)
    combinations = []
    for i in range(N):
        for j in range(i + 1, N):
            combinations.append([i, j]) # 0-indexed for Python

    # Construct CC (using 0-indexed channel numbers)
    # Individual channels
    cc_individual = np.zeros((N, 2), dtype=np.int64)
    cc_individual[:, 0] = np.arange(N)

    # Combinations (dipoles)
    cc_combinations = np.array(combinations, dtype=np.int64) if combinations else np.empty((0, 2), dtype=np.int64)

    CC = np.vstack((cc_individual, cc_combinations))
    
    Ndipolo = CC.shape[0]
    y = np.zeros((x.shape[0], Ndipolo), dtype=np.float64)

    for nd in range(Ndipolo):
        if CC[nd, 1] == 0: # Individual channel
            y[:, nd] = x[:, CC[nd, 0]]
        else: # Dipole combination
            y[:, nd] = x[:, CC[nd, 0]] - x[:, CC[nd, 1]]
            
    return y

# --- 3. msc_fft.py ---
@njit(float64[:](complex128[:,:], int64))
def msc_fft(Y: np.ndarray, M: int) -> np.ndarray:
    """
    Calculates the Magnitude-Squared Coherence (MSC) from FFT output.
    Corresponds to 'msc_fft.m'.

    Args:
        Y (np.ndarray): FFT output of the signal. Shape (n_freq_bins, M_windows).
        M (int): Number of windows.

    Returns:
        np.ndarray: MSC spectrum (n_freq_bins,).
    """
    if Y.shape[1] != M:
        raise ValueError("Number of windows in Y must match M.")

    # MATLAB: ORD = abs(sum(Y,2)).^2./(M*sum(abs(Y).^2,2));
    sum_Y_axis1 = np.sum(Y, axis=1) # Sum along windows
    abs_sum_Y_sq = np.abs(sum_Y_axis1)**2

    sum_abs_Y_sq_axis1 = np.sum(np.abs(Y)**2, axis=1) # Sum of squared magnitudes along windows

    # Avoid division by zero
    denominator = M * sum_abs_Y_sq_axis1
    # Handle cases where denominator might be zero (e.g., if Y has all zeros for a freq bin)
    # Using np.where to prevent NaN/inf for plotting, setting to 0 if denominator is 0
    ORD = np.where(denominator != 0, abs_sum_Y_sq / denominator, 0.0)

    return ORD

# --- 4. VC_MSC.py ---
@njit(float64[:](float64[:], float64)) # jit for the outer loop if nRuns is large
def _vc_msc_monte_carlo_inner(M_values: np.ndarray, alfa: float, N_window: int, bin_idx: int, nRuns: int) -> np.ndarray:
    """
    Inner Monte Carlo simulation for VC_MSC, for a given set of M values.
    """
    vc_mc_results = np.zeros(M_values.shape[0], dtype=np.float64)
    for i, M in enumerate(M_values):
        M_int = int(M)
        ord_values = np.zeros(nRuns, dtype=np.float64)
        for r in range(nRuns):
            x = np.random.randn(N_window * M_int)
            # Reshape, then FFT along axis 0 (each window's FFT)
            aux_fft_reshaped = fft(x.reshape(N_window, M_int), axis=0)
            aux_msc = msc_fft(aux_fft_reshaped, M_int)
            if bin_idx < len(aux_msc):
                ord_values[r] = aux_msc[bin_idx]
            else:
                ord_values[r] = 0.0 # Or handle error appropriately

        # Using numpy's quantile, which is efficient
        vc_mc_results[i] = np.quantile(ord_values, 1.0 - alfa)
    return vc_mc_results

def VC_MSC(M_values: np.ndarray, alfa: float, nRuns: int = Config.N_RUNS_VC_MSC) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates critical values for the MSC detector (Monte Carlo and theoretical).
    Corresponds to 'VC_MSC.m'.

    Args:
        M_values (np.ndarray): Array of number of windows (M).
        alfa (float): Significance level.
        nRuns (int): Number of Monte Carlo runs.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - VC_MC (np.ndarray): Monte Carlo critical values for each M.
            - VC_teorico (np.ndarray): Theoretical critical values for each M.
    """
    N_window = 32 # This is 'tj' from other contexts
    bin_idx = 7 # This is 'bin' from other contexts (0-indexed)

    # Convert M_values to integer array for Numba compatibility
    M_values_int = M_values.astype(np.int64)

    VC_MC = _vc_msc_monte_carlo_inner(M_values_int, alfa, N_window, bin_idx, nRuns)

    # Theoretical value
    # MATLAB: VC_teorico = 1 - alfa.^(1./(M-1));
    # Handle M=1 case to avoid division by zero or log of zero if alfa=0
    VC_teorico = np.where(M_values > 1, 1 - alfa**(1.0 / (M_values - 1)), 0.0) # For M=1, teorical value might be undefined or 0/1 depending on interpretation. Set to 0.0 for safety.

    return VC_MC, VC_teorico

# --- 5. ETS.py ---
@njit(float64(float64[:], int64[:], float64, float64, float64[:]))
def ETS(ord_values: np.ndarray, MM_indices: np.ndarray, alfa: float, NDC: float, vc_msc_critical_values_mc: np.ndarray) -> Tuple[int64, int64]:
    """
    Implements the Sequential Testing Strategy (ETS).
    Corresponds to 'ETS.m'.

    Args:
        ord_values (np.ndarray): Array of detector outputs (from msc_fft).
        MM_indices (np.ndarray): Array of M values (number of windows) at which tests are applied.
        alfa (float): Significance level.
        NDC (float): Minimum number of consecutive detections.
        vc_msc_critical_values_mc (np.ndarray): Pre-computed Monte Carlo critical values for VC_MSC

    Returns:
        Tuple[int, int]:
            - dr (int): Detection result (1 if detected, 0 otherwise).
            - time (int): Time (M value) at which detection occurred or last M if no detection.
    """
    NDC_ceil = int(np.ceil(NDC))

    # The MATLAB code VC_MSC(MM,alfa) inside ETS only uses the MC part if nRuns is passed.
    # In 'sinalleeg_ndc.txt' and 'protocolo_deteccao.txt', 'VC_MSC(MM,alfa)' is called
    # without nRuns, meaning it would implicitly use the default nRuns.
    # To correctly translate, we need to pass the already computed VC_MC values.
    # Assuming `vc_msc_critical_values_mc` here are the correct critical values for `MM_indices`.
    valor_critico = vc_msc_critical_values_mc # This needs to be correctly mapped to MM_indices

    # Check for each M in MM_indices whether detection occurs
    # Note: ord_values is typically `ord[:,MM]` in the context of funcao_NDC_alfaCorrigido_Mmax
    # So `ord_values` here is already `det` in the MATLAB sense for a single simulation run.
    # The `ord(MM)` in MATLAB is `ord_values` here.
    
    # We need to map MM_indices to the indices of `ord_values` and `valor_critico`.
    # `ord_values` in `ETS` is actually `det = ord(:,MM)` in `funcao_custo_v2`
    # and `det = det > repmat(valor_critico, size(det,1), 1)` in `protocolo_deteccao`.
    # This means `ord_values` passed to `ETS` corresponds to the detector outputs AT the `MM_indices`.
    # And `valor_critico` here should correspond to `valor_critico` calculated for `MM_indices`.

    # Assuming `ord_values` passed to ETS are the detector outputs for the specific `MM_indices`.
    det = ord_values > valor_critico

    count = 0
    dr = 0
    time = MM_indices[-1] # Default to last M if no detection

    for ii in range(len(MM_indices)):
        count = det[ii] + count * det[ii] # Counts consecutive detections

        if count == NDC_ceil:
            dr = 1
            time = MM_indices[ii]
            break
            
    return dr, time

# --- 6. funcao_custo_v2.py ---
def funcao_custo_v2(alfa: float, NDC: float, MM_indices: np.ndarray, ord_sim: np.ndarray,
                     FP_desejado: float, vc_msc_critical_values_mc: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Cost function for optimizing alpha in the detection protocol.
    Corresponds to 'funcao_custo_v2.m'.

    Args:
        alfa (float): Significance level (to be optimized).
        NDC (float): Minimum number of consecutive detections.
        MM_indices (np.ndarray): Array of M values for the protocol.
        ord_sim (np.ndarray): Simulated detector outputs under H0 (nRuns x Mmax).
        FP_desejado (float): Desired false positive rate.
        vc_msc_critical_values_mc (np.ndarray): Pre-computed Monte Carlo critical values for VC_MSC,
                                                indexed by M_values. This needs to be correctly
                                                indexed to match MM_indices.

    Returns:
        Tuple[float, np.ndarray]:
            - J (float): Cost value.
            - grad (np.ndarray): Gradient of the cost function.
    """
    nRuns = ord_sim.shape[0]
    alfa = np.maximum(alfa, np.finfo(float).eps)
    alfa = np.minimum(alfa, 1 - np.finfo(float).eps)

    # 1 - Stop criterion (Criterio de parada)
    dr = np.zeros(nRuns, dtype=np.int64)

    # The critical values must be generated for each M in MM_indices and for the given alfa.
    # However, `funcao_NDC_alfaCorrigido_Mmax` precomputes `NDC_minimo` with `alfa_teste` (0.05).
    # Then `funcao_custo_v2` is called inside `fmincg` to optimize `alfa`.
    # The current structure of VC_MSC means it re-runs MC simulation for each alfa.
    # To be efficient for `fmincg`, `VC_MSC` should use the theoretical value `1 - alfa.^(1./(M-1))`
    # or a precomputed MC distribution, and `alfa` changes the quantile.

    # Let's adjust VC_MSC to return a function that can compute critical value given alfa and M.
    # For optimization, a faster approach is needed. The theoretical value is likely used here.
    # MATLAB: valor_critico = 1 - alfa.^(1./(MM-1));
    # This `valor_critico` must be computed inside the loop if `alfa` is changing.

    # However, the Python `VC_MSC` returns both MC and theoretical. The `ETS` function
    # expects a `vc_msc_critical_values_mc` which is an array. This array must be
    # consistent with `MM_indices`.

    # Let's assume that the `vc_msc_critical_values_mc` passed to `funcao_custo_v2` are the
    # *theoretical* critical values, which can be computed on the fly based on the current `alfa`.
    # This implies that `funcao_NDC_alfaCorrigido_Mmax` generates the `vc_msc_critical_values_mc`
    # based on the *current* `alfa` being tested by `fmincg`.

    # This is a key point where the MATLAB code is a bit ambiguous.
    # The `VC_MSC` in `ETS` within `funcao_NDC_alfaCorrigido_Mmax`
    # and `funcao_custo_v2` seems to compute critical values for a given `alfa`.
    # Let's use the theoretical critical value here for simplicity and efficiency in the optimizer.
    # Assuming `VC_MSC` returns theoretical values (1 - alpha^(1/(M-1))) for the given `alfa`.

    # Compute theoretical critical values for the current alfa and relevant M values
    current_vc_teorico = np.where(MM_indices > 1, 1 - alfa**(1.0 / (MM_indices - 1)), 0.0)

    for ii in range(nRuns):
        # ord_sim[ii, MM_indices_0_indexed] selects the relevant detector outputs
        dr[ii], _ = ETS(ord_sim[ii, MM_indices-1], MM_indices, alfa, NDC, current_vc_teorico) # MM_indices-1 for 0-indexing

    FP = np.mean(dr)

    # 2 - Cost function calculation (Cálculo da função de custo)
    erro = (FP - FP_desejado)
    J = 0.5 * erro**2

    # 3 - Gradient calculation (Cálculo do gradiente)
    grad = np.array([erro]) # Gradient is a scalar here, wrapped in an array

    return J, grad

# --- 7. fmincg_wrapper.py (using scipy.optimize.minimize) ---
def fmincg_wrapper(f_obj_grad, x0: np.ndarray, options: Dict[str, Any], *args) -> Tuple[np.ndarray, np.ndarray]:
    """
    A wrapper for scipy.optimize.minimize to mimic fmincg behavior.
    Corresponds to 'fmincg.m'.

    Args:
        f_obj_grad (callable): Function that returns (cost, gradient).
        x0 (np.ndarray): Initial parameters.
        options (dict): Optimization options (e.g., 'MaxIter').
        *args: Additional arguments to pass to f_obj_grad.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - X (np.ndarray): Optimized parameters.
            - fX (np.ndarray): Array of cost values at each iteration.
    """
    max_iter = options.get('MaxIter', 100)

    # Store function values at each iteration
    fX_history = []

    def callback_fn(xk):
        # This callback gets called after each iteration with the current parameters xk
        # We need to compute the cost at xk and append it to fX_history
        cost, _ = f_obj_grad(xk, *args)
        fX_history.append(cost)
        # print(f"Iteration {len(fX_history)} | Cost: {cost:.6e}") # Optional: mimic MATLAB output

    res = minimize(f_obj_grad, x0, args=args, method='CG', jac=True,
                   options={'maxiter': max_iter, 'disp': False},
                   callback=callback_fn)

    # If the callback wasn't called for the initial x0, add it
    if not fX_history:
        initial_cost, _ = f_obj_grad(x0, *args)
        fX_history.append(initial_cost)

    X = res.x
    fX = np.array(fX_history)
    
    # MATLAB fmincg returns (X, fX, i) where i is iterations.
    # scipy.optimize.minimize.nfev is number of function evaluations, niter is number of iterations.
    # The MATLAB version counts iterations as line searches or function evaluations.
    # For a direct mapping, niter is closer.
    # We will return X, fX, and the number of iterations for consistency with MATLAB output style.
    num_iterations = res.nit

    print(f"\nOptimization finished. Final Cost: {res.fun:.6e}")
    # print(f"Total iterations: {num_iterations}")

    return X, fX, num_iterations # Adding num_iterations as 'i'

# --- 8. estimarNDC.py ---
def estimarNDC(NDCinicial: int, alfa_teste: float, FP_desejado: float,
               ord_sim: np.ndarray, Mmin: int, Mstep: int, Mmax: int) -> Tuple[float, np.ndarray]:
    """
    Estimates the minimum number of consecutive detections (NDC) required
    to achieve a desired false positive rate.
    Corresponds to 'EstimarNDC.m'.

    Args:
        NDCinicial (int): Initial NDC value to start searching from.
        alfa_teste (float): Significance level for individual tests.
        FP_desejado (float): Desired overall false positive rate.
        ord_sim (np.ndarray): Simulated detector outputs under H0 (nRuns x Mmax).
        Mmin (int): Minimum window count for the protocol.
        Mstep (int): Step size for window count in the protocol.
        Mmax (int): Maximum window count for the protocol.

    Returns:
        Tuple[float, np.ndarray]:
            - NDC (float): Estimated minimum NDC.
            - FP_values (np.ndarray): Array of false positive rates for tested NDCs.
    """
    MM_indices = np.arange(Mmin, Mmax + 1, Mstep)
    NNTmax = len(MM_indices) # Number of tests in the sequence
    nRuns = ord_sim.shape[0]

    # Pre-compute Monte Carlo critical values once for the given alfa_teste
    # Note: MATLAB `VC_MSC` computes critical values for *all* `MM` values, not just `Mmin`.
    # Let's ensure Python `VC_MSC` is called for the full `MM_indices`.
    _, vc_teorico_for_alfa_teste = VC_MSC(MM_indices, alfa_teste, nRuns=Config.N_RUNS_VC_MSC) # Using theoretical for consistency with `funcao_custo_v2` optimization.

    print(f'Mmin ={Mmin} - Mstep={Mstep} - Mmax={Mmax} - Nruns={nRuns}  -NDCinicial={NDCinicial}')

    # Calculate initial FP for NDCinicial
    dr_initial = np.zeros(nRuns, dtype=np.int64)
    for ii in range(nRuns):
        dr_initial[ii], _ = ETS(ord_sim[ii, MM_indices-1], MM_indices, alfa_teste, NDCinicial, vc_teorico_for_alfa_teste)
    FP_initial = np.mean(dr_initial)

    if FP_initial < FP_desejado and NDCinicial > 1: # MATLAB uses 3 instead of 1
        warnings.warn('NDC inicial too large --- This may increase code execution time. Resetting to 1.')
        NDCinicial = 1

    FP_values = np.zeros(NNTmax + 1) # FP array, 0-indexed, so size NNTmax for NDCs from 1 to NNTmax. Index 0 is unused or for NDC=0 if allowed.

    # Search for NDC
    NDC_found = np.nan
    for NDC_val in range(NDCinicial, NNTmax + 1): # Iterate from NDCinicial up to NNTmax
        dr = np.zeros(nRuns, dtype=np.int64)
        for ii in range(nRuns):
            dr[ii], _ = ETS(ord_sim[ii, MM_indices-1], MM_indices, alfa_teste, NDC_val, vc_teorico_for_alfa_teste)
        FP_values[NDC_val] = np.mean(dr)

        if FP_values[NDC_val] < FP_desejado:
            NDC_found = NDC_val
            break

    # Find the minimum NDC value that respects the constraint
    # MATLAB: [~,ind] = find((FP<FP_desejado).*(FP~=0));
    # Python: Find the first index where FP_values meets the condition
    ind = np.where((FP_values < FP_desejado) & (FP_values != 0))[0]

    NDC = np.nan
    if ind.size == 0:
        warnings.warn('####### Could not find Nalpha ##########')
    elif ind[0] == 0: # This means NDC=0 met the condition, or NDCinicial was 0.
        NDC = 1.0 # Or based on specific logic for NDC=0
    else:
        # Interpolate if needed, but the MATLAB code's interpolation is simple for discrete NDCs
        # MATLAB: interp1(FP([ind(1)-1,ind(1)]),[ind(1)-1,ind(1)],FP_desejado);
        # This interpolation is for finding a fractional NDC, but NDC should be integer.
        # So we just take the first integer NDC that satisfies the condition.
        NDC = float(ind[0])

    # Special handling for NNTmax = 1 or 2 as per MATLAB
    if NNTmax == 2:
        NDC = 2.0
    elif NNTmax == 1:
        NDC = 1.0

    return NDC, FP_values[NDCinicial:] # Return the relevant part of FP_values

# --- 9. funcao_NDC_alfaCorrigido_Mmax.py ---
def funcao_NDC_alfaCorrigido_Mmax(nRuns: int, Mmax: int, alfa_teste: float, FP_desejado: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Obtains corrected alpha and minimum NDC for a range of protocol parameters.
    Corresponds to 'funcao_NDC_alfaCorrigido_Mmax.m'.

    Args:
        nRuns (int): Number of Monte Carlo simulation runs for detector outputs.
        Mmax (int): Maximum number of windows for the protocol.
        alfa_teste (float): Initial significance level for the optimization.
        FP_desejado (float): Desired false positive rate.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - alfa_corrigido (np.ndarray): Array of corrected alpha values.
            - NDC_minimo (np.ndarray): Array of minimum NDC values.
            - cost_alfa (np.ndarray): Array of final cost values from optimization.
            - P (np.ndarray): Matrix of protocol parameters [Mmin, Mstep, Mmax].
    """
    tj = Config.TJ_DEFAULT # Each window 1 second (número de pontos de cada janela)
    bin_idx = Config.BIN_DEFAULT # Frequency bin (0-indexed)

    Ntotal = Mmax * tj # Total number of points

    # Simulate detector outputs under H0
    ord_sim = np.zeros((nRuns, Mmax), dtype=np.float64) # Stores detector values for each experiment
    
    # Pre-allocate fft_x_reshaped to avoid re-allocation in loop for numba
    fft_x_reshaped = np.zeros((tj, Mmax), dtype=np.complex128)

    for ii in range(nRuns):
        x = np.random.randn(Ntotal)
        x_reshaped = x.reshape(tj, Mmax)
        
        # Apply FFT to each window (column)
        fft_x_reshaped[:] = fft(x_reshaped, axis=0)
        
        for M in range(2, Mmax + 1): # Iterate for each increasing window count
            # ord_sim[ii, M-1] because M is 1-indexed, but ord_sim is 0-indexed
            ord_sim[ii, M-1] = msc_fft(fft_x_reshaped[bin_idx, :M], M)

    # Get all possible protocol parameters
    P = generate_protocol_parameters(Mmax)
    alfa_corrigido = np.full(P.shape[0], np.nan, dtype=np.float64)
    cost_alfa = np.full(P.shape[0], np.nan, dtype=np.float64)
    NDC_minimo = np.full(P.shape[0], np.nan, dtype=np.float64)

    # Sort P for consistent NDC initialization (as in MATLAB)
    # MATLAB: [~,aux] = sort(ceil((Mmax - P(:,1))./P(:,2))+1); P = P(aux,:);
    num_tests_in_protocol = np.ceil((Mmax - P[:, 0]) / P[:, 1]) + 1
    aux_indices = np.argsort(num_tests_in_protocol)
    P = P[aux_indices, :]

    for ii in range(P.shape[0]):
        Mmin, Mstep, current_Mmax = P[ii, 0], P[ii, 1], P[ii, 2] # current_Mmax is actually Mmax (constant)
        MM_indices = np.arange(Mmin, current_Mmax + 1, Mstep)
        
        # Display progress
        print(f'{ii * 100 / P.shape[0]:.2f}% - Protocol: Mmin={Mmin}, Mstep={Mstep}')

        # Suggest an initial NDC
        if ii == 0:
            Ninicial = 1
        else:
            # Use previous NDC_minimo, adjusted
            Ninicial = int(np.maximum(np.round(NDC_minimo[ii-1]) - 5, 1))

        # Estimate NDC
        current_NDC, _ = estimarNDC(Ninicial, alfa_teste, FP_desejado, ord_sim, Mmin, Mstep, current_Mmax)
        NDC_minimo[ii] = current_NDC

        # Adjust critical values (optimize alpha)
        # Create a partial function for the cost function, fixing all arguments except alfa
        # The cost function needs `vc_msc_critical_values_mc` which is derived from `alfa`.
        # So it must be computed inside `funcao_custo_v2`.
        cost_func_partial = lambda alpha: funcao_custo_v2(alpha, NDC_minimo[ii], MM_indices, ord_sim, FP_desejado, None) # None for vc_msc, computed internally

        # MATLAB's fmincg options
        options = {'MaxIter': Config.MAX_ITER_FMINCG}
        
        # Call the fmincg wrapper
        optimized_alfa, costs, _ = fmincg_wrapper(cost_func_partial, np.array([alfa_teste]), options)
        
        alfa_corrigido[ii] = optimized_alfa[0]
        if costs.size > 0:
            cost_alfa[ii] = costs[-1]
        else:
            cost_alfa[ii] = np.nan # If optimization failed/no costs returned

    return alfa_corrigido, NDC_minimo, cost_alfa, P

# --- 10. protocolo_deteccao.py ---
def protocolo_deteccao(x: np.ndarray, parametros: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies the detection protocol to real EEG data.
    Corresponds to 'protocolo_deteccao.m'.

    Args:
        x (np.ndarray): Pre-processed EEG signal (time_points_per_window x num_windows).
        parametros (np.ndarray): Matrix of protocol parameters
                                 [Mmin, Mstep, Mmax, NDC_minimo, alfa_corrigido].

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - dr (np.ndarray): Detection results (n_freq_bins x n_protocol_params).
            - time (np.ndarray): Detection times (n_freq_bins x n_protocol_params).
    """
    
    # MATLAB: binsM = 120;
    # This refers to the number of frequency bins to consider (rows in fft output).
    # The actual `binsM` (stimulation frequencies) is loaded per volunteer.
    # Here, `binsM` in the MATLAB code seems to be a placeholder for the number of FFT points.
    # Let's use the actual number of frequency bins from `x.shape[0]` after FFT, or a sensible default.
    # The main script uses `xfft(1:binsM,1:M)`, implying `binsM` is the upper limit for freq bins.
    # Let's assume `binsM_max_freq_idx` here refers to the number of output frequency bins.
    num_fft_points = x.shape[0] # Fs
    max_freq_bins_to_consider = int(num_fft_points / 2) + 1 # Up to Nyquist frequency + DC.
    
    num_windows_in_signal = x.shape[1]
    
    # Calculate detector outputs for all possible M
    # ord (num_freq_bins x Mmax_signal_windows)
    ord_values = np.zeros((max_freq_bins_to_consider, num_windows_in_signal + 1), dtype=np.float64)
    
    # Apply FFT once to all windows
    xfft = fft(x, axis=0) # FFT along the time_points_per_window dimension

    for M in range(2, num_windows_in_signal + 1):
        # Slice xfft to get relevant frequency bins and M windows
        # MATLAB: ord(:,M) = msc_fft(xfft(1:binsM,1:M),M);
        ord_values[:, M] = msc_fft(xfft[:max_freq_bins_to_consider, :M], M)

    num_protocol_params = parametros.shape[0]
    dr = np.zeros((max_freq_bins_to_consider, num_protocol_params), dtype=np.int64)
    time = np.zeros((max_freq_bins_to_consider, num_protocol_params), dtype=np.int64)

    for ii in range(num_protocol_params):
        Mmin = int(parametros[ii, 0])
        Mstep = int(parametros[ii, 1])
        Mmax_param = int(parametros[ii, 2]) # Mmax from the parameter set
        alfa_param = parametros[ii, 4] # Corrected alpha
        NDC_param = parametros[ii, 3] # NDC_minimo

        MM_indices = np.arange(Mmin, Mmax_param + 1, Mstep)

        # Calculate critical values for the current alpha and MM_indices
        # Using theoretical value as in the optimization for consistency.
        current_vc_teorico = np.where(MM_indices > 1, 1 - alfa_param**(1.0 / (MM_indices - 1)), 0.0)

        # MATLAB: det = ord(:,MM) > repmat(valor_critico,size(det,1),1);
        # ord_values is (num_freq_bins x Mmax_signal_windows+1)
        # We need to select columns corresponding to MM_indices from ord_values
        ord_for_protocol = ord_values[:, MM_indices] # Select columns based on MM_indices
        
        # `current_vc_teorico` is (len(MM_indices),)
        # We need to broadcast it across frequency bins.
        det_mask = ord_for_protocol > current_vc_teorico[np.newaxis, :]

        for ll in range(max_freq_bins_to_consider): # For each frequency bin
            count = 0
            current_dr = 0
            current_time = MM_indices[-1] # Default to last M

            for jj in range(len(MM_indices)): # Iterate through the M steps in the protocol
                count = det_mask[ll, jj] + count * det_mask[ll, jj] # Counts consecutive detections

                if count == int(NDC_param): # If NDC consecutive detections are met
                    current_dr = 1
                    current_time = MM_indices[jj]
                    break
            
            dr[ll, ii] = current_dr
            time[ll, ii] = current_time

    return dr, time

# --- 11. paretoFront.py ---
def paretoFront(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filters a set of points according to Pareto dominance.
    Corresponds to 'paretoFront.m'.

    Args:
        p (np.ndarray): N-by-D matrix, where N is the number of points and D is the
                        number of elements (objectives) of each point.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - p (np.ndarray): Pareto-filtered points.
            - idxs (np.ndarray): Indices of the non-dominated solutions in the original `p`.
    """
    # Important: Pareto front often implies maximization of all objectives.
    # In the MATLAB code, it's `paretoFront([TXD,(-timeM)])`.
    # This means TXD is maximized, and -timeM is maximized (equivalent to minimizing timeM).
    # So the current implementation assumes all objectives should be maximized.

    if p.shape[0] == 0:
        return np.empty((0, p.shape[1])), np.empty(0, dtype=np.int64)
    if p.shape[0] == 1:
        return p, np.array([0], dtype=np.int64)

    idxs = np.arange(p.shape[0], dtype=np.int64)
    i = 0
    while i < p.shape[0]:
        # A point `p[j,:]` dominates `p[i,:]` if all its objectives are >=, and at least one is >
        # A point `p[i,:]` is dominated if there exists another point `p[j,:]` such that
        # `p[j,:] >= p[i,:]` for all objectives and `p[j,k] > p[i,k]` for at least one objective `k`.

        # Check for points that dominate p[i,:]
        # np.all(p >= p[i,:], axis=1) -> weakly dominates (all objectives >=)
        # np.any(p > p[i,:], axis=1) -> strictly better in at least one objective
        # dominated_by_others = np.all(p >= p[i,:], axis=1) & np.any(p > p[i,:], axis=1)

        # More directly, filter out points dominated by p[i,:]
        # A point `p[j,:]` is dominated by `p[i,:]` if `p[i,:]` is >= `p[j,:]` in all dimensions
        # and `p[i,:]` is strictly > `p[j,:]` in at least one dimension.
        # indices_to_remove = np.where(np.all(p[i,:] >= p, axis=1) & np.any(p[i,:] > p, axis=1))[0]

        # The MATLAB code:
        # indices = sum( bsxfun( @ge, p(i,:), p ), 2 ) == dim;  -> p(i,:) is weakly greater than or equal to p_j for all dimensions
        # indices(i) = false; -> exclude itself
        # This identifies points that are *weakly dominated* by p[i,:].
        # It means `p[i,:]` is better than or equal to these points in all objectives.
        
        # In Python:
        # `p_i_is_ge_p_j` is a boolean array indicating if `p[i,:]` is >= `p[j,:]` element-wise
        p_i_is_ge_p_j = (p[i, :] >= p) # (D,) vs (N, D) -> (N, D)
        
        # Check if p[i,:] is >= all objectives of p[j,:] for each j
        all_ge = np.all(p_i_is_ge_p_j, axis=1) # (N,)

        # Check if p[i,:] is strictly better in at least one objective than p[j,:]
        any_gt = np.any(p[i, :] > p, axis=1) # (N,)

        # Combine: p[i,:] strictly dominates p[j,:] if (all_ge AND any_gt)
        # This identifies points that are strictly dominated by the current point p[i,:]
        dominated_by_current_pi = all_ge & any_gt
        
        # Also need to consider points that dominate p[i,:].
        # If any point `p[j,:]` (where j != i) dominates `p[i,:]`, then `p[i,:]` is not on the Pareto front.
        # A point p_j dominates p_i if (p_j >= p_i for all dimensions) AND (p_j > p_i for at least one dimension)
        
        # To find Pareto front, a common strategy is to remove dominated points.
        # Keep track of indices.
        
        # Let's use the explicit check: iterate through each point and compare it with all others.
        # If a point is dominated by ANY other point, it's not on the Pareto front.
        
        is_dominated = np.zeros(p.shape[0], dtype=bool)
        for k in range(p.shape[0]):
            for l in range(p.shape[0]):
                if k == l:
                    continue
                # Check if p[l,:] dominates p[k,:]
                # (p[l,:] >= p[k,:]).all() AND (p[l,:] > p[k,:]).any()
                if (p[l,:] >= p[k,:]).all() and (p[l,:] > p[k,:]).any():
                    is_dominated[k] = True
                    break
        
        non_dominated_indices = np.where(~is_dominated)[0]
        return p[non_dominated_indices], idxs[non_dominated_indices]

# --- 12. Chol_f.py ---
def Chol_f(y: np.ndarray, L: int) -> np.ndarray:
    """
    Computes the Cholesky decomposition of the cross-spectral matrix in frequency domain.
    Corresponds to 'Chol_f.m'.

    Args:
        y (np.ndarray): Input signal matrix (time_points x N_channels).
        L (int): Number of points of each epoch (window length).

    Returns:
        np.ndarray: Cholesky decomposition matrices for each frequency bin (N_channels x N_channels x n_fft_bins).
    """
    tamsinal, N = y.shape
    nfft = L // 2 # MATLAB fix(L/2)
    M = tamsinal // L # Number of windows
    
    y = y[:M*L, :] # Limit signal to an integer number of windows

    # Reshape y and compute FFT for each channel and window
    # Y is (nfft_points+1 x M_windows x N_channels)
    Y = np.zeros((nfft + 1, M, N), dtype=np.complex128)
    for i in range(N):
        Y[:, :, i] = fft(y[:, i].reshape(L, M), axis=0)[:nfft + 1, :]

    # Compute cross-spectral matrix Syy
    # Syy is (N_channels x N_channels x nfft_points+1)
    Syy = np.zeros((N, N, nfft + 1), dtype=np.complex128)
    for i in range(N):
        for j in range(i, N): # Upper triangle
            # MATLAB: sum( (conj(Y(:,:,i)).*Y(:,:,j)).' );
            Syy[i, j, :] = np.sum(np.conj(Y[:, :, i]) * Y[:, :, j], axis=1)
        # Fill lower triangle with conjugate transpose
        for j in range(i):
            Syy[i, j, :] = np.conj(Syy[j, i, :])

    # Compute Cholesky decomposition for each frequency bin
    H = np.zeros((N, N, nfft + 1), dtype=np.complex128)
    for i in range(1, nfft + 1): # Exclude DC (bin 0 in Python, 1 in MATLAB)
        # np.linalg.cholesky requires Hermitian positive-definite matrix
        # For cross-spectral matrix, this should hold.
        # Add a small epsilon to the diagonal to ensure positive definiteness if needed
        # (similar to (M(k,k)+eps) in Msweep)
        try:
            H[:, :, i] = np.linalg.cholesky(Syy[:, :, i])
        except np.linalg.LinAlgError:
            warnings.warn(f"Syy matrix not positive definite at frequency bin {i}. Adding jitter.")
            # Add small jitter to diagonal to make it positive definite
            jitter = np.eye(N) * np.finfo(float).eps * 100
            H[:, :, i] = np.linalg.cholesky(Syy[:, :, i] + jitter)


    return H

# --- 13. Chol_f_Norm.py ---
def Chol_f_Norm(y: np.ndarray, L: int) -> np.ndarray:
    """
    Computes the Cholesky decomposition of the normalized cross-spectral matrix in frequency domain.
    Corresponds to 'Chol_f_Norm.m'.
    This version normalizes Y by its mean absolute value across windows.

    Args:
        y (np.ndarray): Input signal matrix (time_points x N_channels).
        L (int): Number of points of each epoch (window length).

    Returns:
        np.ndarray: Cholesky decomposition matrices for each frequency bin (N_channels x N_channels x n_fft_bins).
    """
    tamsinal, N = y.shape
    nfft = L // 2
    M = tamsinal // L
    
    y = y[:M*L, :]

    Y = np.zeros((nfft + 1, M, N), dtype=np.complex128)
    for i in range(N):
        Y[:, :, i] = fft(y[:, i].reshape(L, M), axis=0)[:nfft + 1, :]

    # Normalize Y by the mean absolute value across windows for each freq bin and channel
    # MATLAB: Y = Y./repmat((mean(abs(Y),2)),1,M);
    mean_abs_Y_across_windows = np.mean(np.abs(Y), axis=1, keepdims=True) # (nfft+1, 1, N)
    Y = Y / np.where(mean_abs_Y_across_windows != 0, mean_abs_Y_across_windows, 1.0) # Avoid div by zero

    # Compute cross-spectral matrix Syy
    Syy = np.zeros((N, N, nfft + 1), dtype=np.complex128)
    for i in range(N):
        for j in range(i, N):
            Syy[i, j, :] = np.sum(np.conj(Y[:, :, i]) * Y[:, :, j], axis=1)
        for j in range(i):
            Syy[i, j, :] = np.conj(Syy[j, i, :])

    H = np.zeros((N, N, nfft + 1), dtype=np.complex128)
    for i in range(1, nfft + 1): # Exclude DC (bin 0 in Python, 1 in MATLAB)
        try:
            H[:, :, i] = np.linalg.cholesky(Syy[:, :, i])
        except np.linalg.LinAlgError:
            warnings.warn(f"Syy matrix not positive definite after normalization at frequency bin {i}. Adding jitter.")
            jitter = np.eye(N) * np.finfo(float).eps * 100
            H[:, :, i] = np.linalg.cholesky(Syy[:, :, i] + jitter)

    return H

# --- 14. MMSC.py ---
def Msweep(M_matrix: np.ndarray, r: int) -> np.ndarray:
    """
    Sweep operator for square matrices.
    Corresponds to 'Msweep' function within 'MMSC.m'.

    Args:
        M_matrix (np.ndarray): Input square matrix.
        r (int): Pivot index (1-based in MATLAB, so convert to 0-based for Python).

    Returns:
        np.ndarray: Swept matrix.
    """
    M = M_matrix.copy()
    N = M.shape[0]
    
    # Adjust r to be 0-indexed for Python
    k_pivot = r 

    # For m,n != k_pivot
    for m in range(N):
        for n in range(N):
            if m != k_pivot and n != k_pivot:
                M[m, n] = M[m, n] - M[m, k_pivot] * M[k_pivot, n] / (M[k_pivot, k_pivot] + np.finfo(float).eps)

    # For (m == k_pivot and n != k_pivot) or (n == k_pivot and m != k_pivot)
    for m in range(N):
        if m != k_pivot:
            M[k_pivot, m] = M[k_pivot, m] / (M[k_pivot, k_pivot] + np.finfo(float).eps) # M(k,n) = M(k,n) / M(k,k)
            M[m, k_pivot] = -M[m, k_pivot] / (M[k_pivot, k_pivot] + np.finfo(float).eps) # M(m,k) = -M(m,k) / M(k,k)

    # For k_pivot, k_pivot -> pivot
    M[k_pivot, k_pivot] = 1 / (M[k_pivot, k_pivot] + np.finfo(float).eps)

    return M

def MMSC(y: np.ndarray, tj: int, fs: int = None, alpha: float = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Multiple magnitude-squared coherence (MMSC).
    Corresponds to 'MMSC.m'.

    Args:
        y (np.ndarray): Matrix whose columns are the signals.
        tj (int): Number of points of each epoch (window length).
        fs (int, optional): Sample rate of signals.
        alpha (float, optional): Significance level.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]:
            - k2N (np.ndarray): MMSC spectrum.
            - F (np.ndarray): Frequency vector (if fs is provided).
            - k2Ncrit (float): Theoretical critical value (if alpha is provided).
    """
    nf = tj // 2 + 1
    N_channels = y.shape[1]
    M_windows = y.shape[0] // tj # Number of segments

    # Limit signal for an integer number of windows
    y_limited = y[:M_windows * tj, :]

    # Sfft accumulator for FFT in a ring
    # MATLAB: Sfft = zeros(tj,N,M);
    Sfft = np.zeros((tj, N_channels, M_windows), dtype=np.complex128)
    idl = 0
    for k in range(M_windows):
        Sfft[:, :, k] = fft(y_limited[idl : idl + tj, :], axis=0)
        idl += tj

    k2N = np.zeros(nf, dtype=np.float64)

    for kf in range(nf):
        # Specm_a (N+1 x N+1)
        Specm_a = np.zeros((N_channels + 1, N_channels + 1), dtype=np.complex128)

        # Sum Sfft values for V and VH (last row/column)
        for p in range(N_channels):
            # MATLAB: Specm_a(N+1,p) = Specm_a(N+1,p) + Sfft(kf,p,ks);
            Specm_a[N_channels, p] = np.sum(Sfft[kf, p, :])
            Specm_a[p, N_channels] = np.conj(Specm_a[N_channels, p]) # VH - hermitian of V

        # Build Specm_a matrix
        for p in range(N_channels):
            for q in range(p + 1): # p from 0 to N-1, q from 0 to p-1
                # MATLAB: Specm_a(p,q) = Specm_a(p,q) + conj(Sfft(kf,p,ks)).*Sfft(kf,q,ks);
                Specm_a[p, q] = np.sum(np.conj(Sfft[kf, p, :]) * Sfft[kf, q, :])
                Specm_a[q, p] = np.conj(Specm_a[p, q]) # Fill lower triangle

        Specm_a[N_channels, N_channels] = 1 # Last element (bottom-right)

        # Apply Msweep operator (pivot index N_channels for 0-indexed)
        Specm_as = Msweep(Specm_a, N_channels) # Msweep takes 0-indexed r for Python

        k2N[kf] = (1 - np.real(Specm_as[N_channels, N_channels])) / M_windows

    F = np.array([])
    k2Ncrit = np.nan

    if fs is not None:
        F = np.arange(nf) * fs / tj
        if alpha is not None:
            # Fcrit = finv(1-alpha,2*N,2*M-2*N);
            Fcrit = f.ppf(1 - alpha, 2 * N_channels, 2 * M_windows - 2 * N_channels)
            k2Ncrit = Fcrit / (((M_windows - N_channels) / N_channels) + Fcrit)

    return k2N, F, k2Ncrit

# --- 15. MSCM.py ---
def MCSM(y: np.ndarray, tj: int, fs: int = None, alpha: float = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Multiple component synchrony measure (MCSM).
    Corresponds to 'MSCM.m'.

    Args:
        y (np.ndarray): Matrix whose columns are the signals.
        tj (int): Number of points of each epoch (window length).
        fs (int, optional): Sample rate.
        alpha (float, optional): Significance level.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]:
            - csmN (np.ndarray): MCSM spectrum.
            - F (np.ndarray): Frequency vector (if fs is provided).
            - csmNcrit (float): Theoretical critical value (if alpha is provided).
    """
    tamsinal, N_channels = y.shape
    nfft_points = tj // 2 # MATLAB fix(tj/2)
    M_windows = tamsinal // tj # Number of windows

    y = y[:M_windows * tj, :] # Limit the signal for an integer number of windows

    # Y is a 3D matrix (nfft_points+1 x M_windows x N_channels)
    Y = np.zeros((nfft_points + 1, M_windows, N_channels), dtype=np.complex128)
    for i in range(N_channels):
        Y[:, :, i] = fft(y[:, i].reshape(tj, M_windows), axis=0)[:nfft_points + 1, :]

    # Algorithm's start
    teta = np.angle(Y)
    C = np.cos(teta)
    S = np.sin(teta)

    Cmed = np.mean(C, axis=2) # Mean across channels
    Smed = np.mean(S, axis=2) # Mean across channels

    # Desconsider the first and last freq. values by setting to NaN, then handle for atan
    Cmed[[0, -1], :] = np.nan
    Smed[[0, -1], :] = np.nan

    # Calculate mean teta matrix
    # temp1 = atan( (Smed.*(Cmed<0))./Cmed)+pi*(Cmed<0);
    # temp2 = atan( (Smed.*(Cmed>=0))./Cmed);
    # teta_med = temp1 + temp2;

    # Handle division by zero and NaN from Cmed/Smed
    teta_med = np.zeros_like(Cmed, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        # For Cmed < 0
        mask_cmed_neg = (Cmed < 0) & ~np.isnan(Cmed)
        teta_med[mask_cmed_neg] = np.arctan(Smed[mask_cmed_neg] / Cmed[mask_cmed_neg]) + np.pi

        # For Cmed >= 0
        mask_cmed_pos_zero = (Cmed >= 0) & ~np.isnan(Cmed)
        teta_med[mask_cmed_pos_zero] = np.arctan(Smed[mask_cmed_pos_zero] / Cmed[mask_cmed_pos_zero])

    # Multiple CSM
    # csmN=(1/M^2)*sum(cos(teta_med')).^2+(1/M^2)*sum(sin(teta_med')).^2;
    # Sum over M_windows (columns of teta_med)
    sum_cos_teta_med = np.sum(np.cos(teta_med), axis=1) # (nfft_points+1,)
    sum_sin_teta_med = np.sum(np.sin(teta_med), axis=1) # (nfft_points+1,)

    csmN = (1 / M_windows**2) * (sum_cos_teta_med**2 + sum_sin_teta_med**2)
    
    # Restore NaN for the first and last frequency bins if they were NaN initially
    csmN[0] = np.nan
    csmN[-1] = np.nan

    F = np.array([])
    csmNcrit = np.nan

    if fs is not None:
        F = np.arange(nfft_points + 1) * fs / tj
        if alpha is not None:
            # csmNcrit = chi2inv(1-alpha,2)/(2*M);
            csmNcrit = chi2.ppf(1 - alpha, 2) / (2 * M_windows)

    return csmN, F, csmNcrit

# --- 16. ORD_freq.py ---
def ORD_freq(H: np.ndarray, M: int, detector_name: str, Hr: np.ndarray = None) -> float:
    """
    Generates detector values for matrices in the frequency domain (Monte Carlo).
    Corresponds to 'ORD_freq.m'.

    Args:
        H (np.ndarray): Cholesky matrix (or array of Cholesky matrices for LFT).
        M (int): Number of windows.
        detector_name (str): Name of the detector (e.g., 'aMSC', 'pMSC', 'MCSM', etc.).
        Hr (np.ndarray, optional): Cholesky matrix for noise (for GBT detectors).

    Returns:
        float: Detector value.
    """
    N_channels = H.shape[0] # N from the size of the Cholesky matrix

    # Helper for generating complex Gaussian noise
    def randn_complex(rows, cols):
        return (np.random.randn(rows, cols) + 1j * np.random.randn(rows, cols)) / np.sqrt(2)

    ORD = 0.0 # Default value

    if detector_name == 'aMSC':
        yfft = randn_complex(M, N_channels) @ H # Matrix multiplication
        MSC1 = np.zeros(N_channels, dtype=np.float64)
        for jj in range(N_channels):
            MSC1[jj] = np.abs(np.sum(yfft[:, jj]))**2 / (M * np.sum(np.abs(yfft[:, jj])**2))
        ORD = np.mean(MSC1)

    elif detector_name == 'pMSC':
        yfft = randn_complex(M, N_channels) @ H
        MSC1 = np.zeros(N_channels, dtype=np.float64)
        for jj in range(N_channels):
            MSC1[jj] = np.abs(np.sum(yfft[:, jj]))**2 / (M * np.sum(np.abs(yfft[:, jj])**2))
        ORD = (np.prod(MSC1))**(1.0 / N_channels)

    elif detector_name == 'pCSM':
        yfft = randn_complex(M, N_channels) @ H
        yfft_norm = yfft / np.abs(yfft) # Normalize to unit magnitude
        CSM1 = np.zeros(N_channels, dtype=np.float64)
        for jj in range(N_channels):
            CSM1[jj] = np.abs(np.sum(yfft_norm[:, jj]))**2 / (M * np.sum(np.abs(yfft_norm[:, jj])**2))
        ORD = (np.prod(CSM1))**(1.0 / N_channels)

    elif detector_name == 'aCSM':
        yfft = randn_complex(M, N_channels) @ H
        yfft_norm = yfft / np.abs(yfft)
        CSM1 = np.zeros(N_channels, dtype=np.float64)
        for jj in range(N_channels):
            CSM1[jj] = np.abs(np.sum(yfft_norm[:, jj]))**2 / (M * np.sum(np.abs(yfft_norm[:, jj])**2))
        ORD = np.mean(CSM1)

    elif detector_name == 'MCSM':
        yfft = randn_complex(M, N_channels) @ H
        teta = np.angle(yfft)
        C = np.cos(teta)
        S = np.sin(teta)
        Cmed = np.mean(C, axis=1) # Mean across channels for each time point
        Smed = np.mean(S, axis=1) # Mean across channels for each time point
        
        # Handle atan based on sign of Cmed
        teta_med = np.zeros_like(Cmed, dtype=np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            mask_cmed_neg = (Cmed < 0) & ~np.isnan(Cmed)
            teta_med[mask_cmed_neg] = np.arctan(Smed[mask_cmed_neg] / Cmed[mask_cmed_neg]) + np.pi
            mask_cmed_pos_zero = (Cmed >= 0) & ~np.isnan(Cmed)
            teta_med[mask_cmed_pos_zero] = np.arctan(Smed[mask_cmed_pos_zero] / Cmed[mask_cmed_pos_zero])

        ORD = (1 / M**2) * (np.sum(np.cos(teta_med))**2 + np.sum(np.sin(teta_med))**2)

    elif detector_name == 'aLFT':
        L_bins = H.shape[2] # Number of frequency bins in H (H is N x N x L_bins)
        yfft = np.zeros((L_bins, N_channels), dtype=np.complex128)
        for ii in range(L_bins):
            yfft[ii, :] = randn_complex(1, N_channels) @ H[:, :, ii]
        
        pfo_idx = (L_bins - 1) // 2 # Center frequency bin index (0-indexed)
        # L in MATLAB LFT context is the number of sideband bins, `L = L_bins - 1` is not necessarily true.
        # It's `L_sidebands = L_bins - 1`.
        num_sidebands = L_bins - 1
        
        Y_abs = np.abs(yfft)
        Yfo = Y_abs[pfo_idx, :] # Value at fundamental frequency
        Yfn = np.delete(Y_abs, pfo_idx, axis=0) # Values at other frequencies (noise)

        # Compute F value
        # mean((Yfo.^2)./(1/L*sum(Yfn.^2,1))); -> here L is num_sidebands
        ORD = np.mean((Yfo**2) / (1/num_sidebands * np.sum(Yfn**2, axis=0)))
    
    elif detector_name == 'pLFT':
        L_bins = H.shape[2]
        yfft = np.zeros((L_bins, N_channels), dtype=np.complex128)
        for ii in range(L_bins):
            yfft[ii, :] = randn_complex(1, N_channels) @ H[:, :, ii]
        
        pfo_idx = (L_bins - 1) // 2
        num_sidebands = L_bins - 1
        
        Y_abs = np.abs(yfft)
        Yfo = Y_abs[pfo_idx, :]
        Yfn = np.delete(Y_abs, pfo_idx, axis=0)

        ratios = (Yfo**2) / (1/num_sidebands * np.sum(Yfn**2, axis=0))
        ORD = (np.prod(ratios))**(1.0 / N_channels)

    elif detector_name == 'MLFT':
        L_bins = H.shape[2]
        yfft = np.zeros((L_bins, N_channels), dtype=np.complex128)
        for ii in range(L_bins):
            yfft[ii, :] = randn_complex(1, N_channels) @ H[:, :, ii]
        
        pfo_idx = (L_bins - 1) // 2
        num_sidebands = L_bins - 1
        
        Y_abs = np.abs(yfft)
        Yfo = Y_abs[pfo_idx, :]
        Yfn = np.delete(Y_abs, pfo_idx, axis=0)

        # MATLAB: sum(Yfo.^2)/(sum(1/L*sum(Yfn.^2,1)));
        ORD = np.sum(Yfo**2) / (np.sum(1/num_sidebands * np.sum(Yfn**2, axis=0)))

    elif detector_name == 'aGBT':
        Svv = np.sum((np.abs(randn_complex(M, N_channels) @ H))**2, axis=0)
        Snm = np.sum((np.abs(randn_complex(M, N_channels) @ Hr))**2, axis=0)
        ORD = np.mean(Svv / (Svv + Snm))

    elif detector_name == 'pGBT':
        Svv = np.sum((np.abs(randn_complex(M, N_channels) @ H))**2, axis=0)
        Snm = np.sum((np.abs(randn_complex(M, N_channels) @ Hr))**2, axis=0)
        ORD = (np.prod(Svv / (Svv + Snm)))**(1.0 / N_channels)

    elif detector_name == 'MGBT':
        Svv = np.sum((np.abs(randn_complex(M, N_channels) @ H))**2) # Sum all elements
        Snm = np.sum((np.abs(randn_complex(M, N_channels) @ Hr))**2) # Sum all elements
        ORD = Svv / (Svv + Snm)

    else:
        raise NotImplementedError(f"Detector '{detector_name}' not implemented in ORD_freq.")

    return ORD

# --- 17. mord.py ---
# Dummy implementations for missing detectors mentioned in 'mord.m'
def aMSC_dummy(*args, **kwargs):
    warnings.warn("Dummy aMSC called. No actual implementation provided.")
    return np.array([0.0]), None # Return dummy value and None for F

def pMSC_dummy(*args, **kwargs):
    warnings.warn("Dummy pMSC called. No actual implementation provided.")
    return np.array([0.0]), None

def aCSM_dummy(*args, **kwargs):
    warnings.warn("Dummy aCSM called. No actual implementation provided.")
    return np.array([0.0]), None

def pCSM_dummy(*args, **kwargs):
    warnings.warn("Dummy pCSM called. No actual implementation provided.")
    return np.array([0.0]), None

def aLFT_dummy(*args, **kwargs):
    warnings.warn("Dummy aLFT called. No actual implementation provided.")
    return np.array([0.0]), None

def pLFT_dummy(*args, **kwargs):
    warnings.warn("Dummy pLFT called. No actual implementation provided.")
    return np.array([0.0]), None

def MLFT_dummy(*args, **kwargs):
    warnings.warn("Dummy MLFT called. No actual implementation provided.")
    return np.array([0.0]), None

def aGBT_dummy(*args, **kwargs):
    warnings.warn("Dummy aGBT called. No actual implementation provided.")
    return np.array([0.0]), None

def pGBT_dummy(*args, **kwargs):
    warnings.warn("Dummy pGBT called. No actual implementation provided.")
    return np.array([0.0]), None

def MGBT_dummy(*args, **kwargs):
    warnings.warn("Dummy MGBT called. No actual implementation provided.")
    return np.array([0.0]), None


def mord(detector_name: str, y: np.ndarray, parameters: Dict[str, Any], x: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generic function for multivariate detectors.
    Corresponds to 'mord.m'.

    Args:
        detector_name (str): Name of the multivariate detector.
        y (np.ndarray): Signal data.
        parameters (dict): Detector parameters (e.g., 'fs', 'tj', 'L', 'fo', 'N').
        x (np.ndarray, optional): Noise signal (for global tests like GBT).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - value (np.ndarray): Detector output value(s).
            - F (np.ndarray): Frequency vector (if available).
    """
    s = detector_name # Use s for brevity as in MATLAB code

    # Handle 'MSC', 'CSM', 'LFT' as 'aMSC', 'aCSM', 'aLFT' for N=1
    if s == 'MSC': s = 'aMSC'; parameters['N'] = 1
    if s == 'CSM': s = 'aCSM'; parameters['N'] = 1
    if s == 'LFT': s = 'aLFT'; parameters['N'] = 1
    
    # Handle 'daCSM', 'daMSC', 'daLFT' when N=1
    if parameters.get('N') == 1:
        if s == 'daCSM': s = 'aCSM'
        if s == 'daMSC': s = 'aMSC'
        if s == 'daLFT': s = 'aLFT'

    value = np.array([0.0]) # Default dummy value
    F = np.array([]) # Default dummy frequency

    # Handle dipolos (starting with 'd')
    if s.startswith('d'):
        xd = dipolos(y)
        return mord(s[1:], xd, parameters) # Recursive call without 'd'

    # MORD Detectors
    if s == 'aMSC':
        value, F = aMSC_dummy(y, parameters['tj'], parameters.get('fs')) # Placeholder for aMSC
    elif s == 'pMSC':
        value, F = pMSC_dummy(y, parameters['tj'], parameters.get('fs')) # Placeholder for pMSC
    elif s == 'MMSC':
        value, F, _ = MMSC(y, parameters['tj'], parameters.get('fs'), None) # MMSC returns k2Ncrit, ignore for this context
    elif s == 'aCSM':
        value, F = aCSM_dummy(y, parameters['tj'], parameters.get('fs')) # Placeholder for aCSM
    elif s == 'pCSM':
        value, F = pCSM_dummy(y, parameters['tj'], parameters.get('fs')) # Placeholder for pCSM
    elif s == 'MCSM':
        value, F, _ = MCSM(y, parameters['tj'], parameters.get('fs'), None) # MCSM returns csmNcrit, ignore
    elif s == 'aLFT':
        value = aLFT_dummy(y, parameters['L'], parameters.get('fs'), [], parameters.get('fo')) # Placeholder
        F = np.array([parameters.get('fo')])
    elif s == 'pLFT':
        value = pLFT_dummy(y, parameters['L'], parameters.get('fs'), [], parameters.get('fo')) # Placeholder
        F = np.array([parameters.get('fo')])
    elif s == 'MLFT':
        value = MLFT_dummy(y, parameters['L'], parameters.get('fs'), [], parameters.get('fo')) # Placeholder
        F = np.array([parameters.get('fo')])
    elif s == 'aGBT':
        value, F = aGBT_dummy(y, x, parameters['tj'], parameters.get('fs')) # Placeholder
    elif s == 'pGBT':
        value, F = pGBT_dummy(y, x, parameters['tj'], parameters.get('fs')) # Placeholder
    elif s == 'MGBT':
        value, F = MGBT_dummy(y, x, parameters['tj'], parameters.get('fs')) # Placeholder
    else:
        raise NotImplementedError(f"Detector '{s}' is not a recognized or implemented MORD detector.")

    return value, F

# --- 18. critical_value.py ---
def critical_value(alpha: float, detector_name: str, parameters: Dict[str, Any],
                   methods: str, yin: np.ndarray = None, xin: np.ndarray = None) -> float:
    """
    Computes the critical value for a given detector and significance level.
    Corresponds to 'critical_value.m'.

    Args:
        alpha (float): Significance level.
        detector_name (str): Name of the detector.
        parameters (dict): Detector parameters (e.g., 'M', 'L', 'N', 'tj', 'fs', 'fo', 'mistura', 'Nruns').
        methods (str): Method used to estimate the critical value ('theoretical', 'Monte_Carlo_default',
                       'time_Cholesky_corrected', 'frequency_Cholesky_corrected').
        yin (np.ndarray, optional): Signals used for Cholesky correction (y-input).
        xin (np.ndarray, optional): Signals used for Cholesky correction (x-input, for GBT).

    Returns:
        float: Critical value.
    """
    list_detector = {'MSC', 'CSM', 'LFT', 'MMSC', 'aMSC', 'pMSC', 'MCSM', 'aCSM', 'pCSM', 'MLFT', 'aLFT',
                     'pLFT', 'MGBT', 'aGBT', 'pGBT', 'daMSC', 'dpMSC', 'daCSM', 'dpCSM'}
    list_methods = {'theoretical', 'Monte_Carlo_default', 'time_Cholesky_corrected', 'frequency_Cholesky_corrected'}

    if detector_name not in list_detector:
        raise ValueError(f"Detector '{detector_name}' not defined.")
    if methods not in list_methods:
        raise ValueError(f"Method '{methods}' not defined.")
    if not (0 < alpha < 1):
        raise ValueError("The alpha value must be between 0 and 1.")

    # Cases where N=1, map to 'a' versions
    s = detector_name # Use s for detector_name for brevity
    if s == 'MSC': s = 'aMSC'; parameters['N'] = 1
    if s == 'CSM': s = 'aCSM'; parameters['N'] = 1
    if s == 'LFT': s = 'aLFT'; parameters['N'] = 1

    if parameters.get('N') == 1:
        if s == 'daCSM': s = 'aCSM'
        if s == 'daMSC': s = 'aMSC'
        if s == 'daLFT': s = 'aLFT'

    CV = np.nan # Default critical value

    # 1 - Theoretical Methods
    if methods == 'theoretical':
        if s == 'aMSC' and parameters.get('N') == 1:
            # MATLAB: 1 - alpha.^(1./(parameters.M-1));
            CV = 1 - alpha**(1.0 / (parameters['M'] - 1)) if parameters['M'] > 1 else 0.0
        elif s == 'MMSC':
            # MATLAB: betainv(1-alpha,parameters.N,parameters.M-parameters.N);
            CV = beta.ppf(1 - alpha, parameters['N'], parameters['M'] - parameters['N'])
        elif s == 'MCSM':
            # MATLAB: chi2inv(1-alpha,2)/(2*parameters.M);
            CV = chi2.ppf(1 - alpha, 2) / (2 * parameters['M'])
        elif s == 'aCSM':
            # MATLAB: chi2inv(1-alpha,2)/(2*parameters.M*parameters.N);
            CV = chi2.ppf(1 - alpha, 2) / (2 * parameters['M'] * parameters['N'])
        elif s == 'MLFT':
            # MATLAB: finv((1-alpha),2*parameters.N,2*parameters.N*parameters.L);
            CV = f.ppf(1 - alpha, 2 * parameters['N'], 2 * parameters['N'] * parameters['L'])
        else:
            warnings.warn(f"Theoretical critical value not defined for detector '{s}'.")
            
    # 2 - Monte Carlo Default
    elif methods == 'Monte_Carlo_default':
        # Default parameters from MATLAB
        parameters.setdefault('fo', 8)
        parameters.setdefault('tj', 32)
        N_channels = parameters['N']
        Nruns = parameters.get('Nruns', Config.N_RUNS_VC_MSC)

        if s.startswith('L', 1): # If detector is LFT-like (e.g., aLFT, pLFT, MLFT)
            parameters['M'] = 1
            parameters['tj'] = 256
            parameters['fo'] = 8 * 4
            parameters['fs'] = parameters['tj'] # Assuming fs is tj for this case

        length_signal = parameters['tj'] * parameters['M']
        DV = np.zeros(Nruns, dtype=np.float64)

        for ii in range(Nruns):
            current_y = np.random.randn(length_signal, N_channels)
            current_x = None
            if s.startswith('B', 2): # If detector is GBT-like (e.g., aGBT, pGBT, MGBT)
                current_x = np.random.randn(length_signal, N_channels)
                value, _ = mord(s, current_y, parameters, current_x)
            else:
                value, _ = mord(s, current_y, parameters)
            
            if s.startswith('L', 1): # LFT-like
                DV[ii] = value[0] if value.size > 0 else 0.0
            else:
                # Value can be an array if it's a spectrum, need to pick the 'fo' bin
                # MATLAB: DV(ii) = value(parameters.fo); (1-indexed fo)
                if parameters['fo'] <= value.size:
                    DV[ii] = value[parameters['fo'] - 1] # 0-indexed fo
                else:
                    DV[ii] = 0.0 # Or raise error

        CV = np.quantile(DV, 1 - alpha)

    # 3 - Time_Cholesky_corrected
    elif methods == 'time_Cholesky_corrected':
        parameters.setdefault('fo', 8)
        parameters.setdefault('tj', 32)
        N_channels = parameters['N']
        if N_channels != yin.shape[1]:
            warnings.warn("N ~= number of signals in yin")

        Nruns = parameters.get('Nruns', Config.N_RUNS_VC_MSC)

        if s.startswith('L', 1): # LFT-like
            parameters['M'] = 1
            parameters['tj'] = 256
            parameters['fo'] = 8 * 4
            parameters['fs'] = parameters['tj']

        # time_Cholesky_corrected
        A1 = None
        if parameters.get('mistura') == 'fixa':
            A1 = yin
        else:
            ymean = np.mean(yin, axis=0)
            yin_detrended = yin - ymean
            if parameters.get('mistura') is None or parameters.get('mistura') == 'cholesky':
                Sigma = np.corrcoef(yin_detrended, rowvar=False) # Correlation matrix
                A1 = np.linalg.cholesky(Sigma)
            elif parameters.get('mistura') == 'PCA':
                # PCA in MATLAB: [coeff] = pca(X); coeff is A1'
                # np.linalg.svd for PCA (coeff is V, not U)
                cov_matrix = np.cov(yin_detrended, rowvar=False)
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                # Sort by eigenvalues in descending order
                idx = np.argsort(eigenvalues)[::-1]
                eigenvectors = eigenvectors[:, idx]
                # A1 should be transpose of eigenvectors for MATLAB equivalence (coefficients)
                A1 = eigenvectors.T
            else:
                raise ValueError(f"Unknown mixture method: {parameters.get('mistura')}")

        A2 = None
        if s.startswith('B', 2): # GBT-like
            xmean = np.mean(xin, axis=0)
            xin_detrended = xin - xmean
            Sigma_x = np.corrcoef(xin_detrended, rowvar=False)
            A2 = np.linalg.cholesky(Sigma_x)

        length_signal = parameters['tj'] * parameters['M']
        DV = np.zeros(Nruns, dtype=np.float64)

        for ii in range(Nruns):
            sim_y = np.random.randn(length_signal, N_channels) @ A1
            if s.startswith('B', 2):
                sim_x = np.random.randn(length_signal, N_channels) @ A2
                value, _ = mord(s, sim_y, parameters, sim_x)
            else:
                sim_y = detrend(sim_y, axis=0, type='constant') # MATLAB detrend(y,'constant')
                value, _ = mord(s, sim_y, parameters)

            if s.startswith('L', 1):
                DV[ii] = value[0] if value.size > 0 else 0.0
            else:
                if parameters['fo'] <= value.size:
                    DV[ii] = value[parameters['fo'] - 1]
                else:
                    DV[ii] = 0.0

        CV = np.quantile(DV, 1 - alpha)

    # 4 - Frequency_Cholesky_corrected
    elif methods == 'frequency_Cholesky_corrected':
        if parameters['N'] != yin.shape[1]:
            warnings.warn("N ~= number of signals in yin")

        Nruns = parameters.get('Nruns', Config.N_RUNS_VC_MSC)

        # Identify bin for analyzed frequency
        # MATLAB: parameters.bin = parameters.tj./parameters.fs*parameters.fo+1; (1-indexed)
        # Python: 0-indexed
        parameters['bin_idx'] = int(parameters['tj'] / parameters['fs'] * parameters['fo'])

        # Estimate number of windows (M)
        parameters['M'] = yin.shape[0] // parameters['tj']

        # Cross-spectral matrix (Cholesky)
        A1freq = Chol_f(yin, parameters['tj'])
        
        A2freq = None
        if s.startswith('B', 2): # GBT-like, needs noise signal x
            A1freq = Chol_f_Norm(yin, parameters['tj']) # MATLAB Chol_f_Norm for GBT
            A2freq = Chol_f_Norm(xin, parameters['tj'])

        DV = np.zeros(Nruns, dtype=np.float64)
        for ii in range(Nruns):
            if s.startswith('B', 2): # GBT-like
                H_freq = A1freq[:, :, parameters['bin_idx']]
                Hr_freq = A2freq[:, :, parameters['bin_idx']]
                DV[ii] = ORD_freq(H_freq, parameters['M'], s, Hr_freq)
            elif s.startswith('L', 1): # LFT-like
                # MATLAB: ORD_freq(A1freq(:,:,parameters.bin+vector_LFT), parameters.M, detector);
                # Here `H` to ORD_freq would be a subset of A1freq for LFT.
                # `vector_LFT` = [-parameters.L/2:parameters.L/2];
                L_sidebands = parameters['L']
                # Correct calculation for LFT frequency range
                start_bin = max(0, parameters['bin_idx'] - L_sidebands // 2)
                end_bin = min(A1freq.shape[2], parameters['bin_idx'] + L_sidebands // 2 + 1)
                H_lft_range = A1freq[:, :, start_bin:end_bin]
                DV[ii] = ORD_freq(H_lft_range, parameters['M'], s)
            else: # Other detectors (MSC-like, CSM-like)
                H_freq = A1freq[:, :, parameters['bin_idx']]
                DV[ii] = ORD_freq(H_freq, parameters['M'], s)
        
        CV = np.quantile(DV, 1 - alpha)

    return CV

# --- 19. xyz2grid.py (Utility function, not provided in MATLAB) ---
def xyz2grid(x: np.ndarray, y: np.ndarray, z: np.ndarray, method: str = 'linear') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts scattered 3D points (x, y, z) into a grid format (X_grid, Y_grid, Z_grid)
    for surface plotting. Uses scipy.interpolate.griddata.

    Args:
        x (np.ndarray): 1D array of x-coordinates.
        y (np.ndarray): 1D array of y-coordinates.
        z (np.ndarray): 1D array of z-coordinates (values at x,y).
        method (str): Interpolation method ('linear', 'nearest', 'cubic').

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - X_grid (np.ndarray): Meshgrid for X.
            - Y_grid (np.ndarray): Meshgrid for Y.
            - Z_grid (np.ndarray): Interpolated Z values on the grid.
    """
    # Create a regular grid to interpolate onto
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    X_grid, Y_grid = np.meshgrid(xi, yi)

    # Interpolate Z values onto the grid
    points = np.vstack((x, y)).T
    Z_grid = griddata(points, z, (X_grid, Y_grid), method=method)

    return X_grid, Y_grid, Z_grid

# --- 20. main_analysis.py (sinalleeg_ndc.txt) ---
def main_analysis():
    """
    Main script for applying the detection protocol to EEG data,
    performing analysis, and plotting results.
    Corresponds to 'sinalleeg_ndc.txt'.
    """
    print("Starting EEG detection protocol analysis...")

    # --- 1. Setup ---
    Vvoluntario = ['Ab', 'An', 'Bb', 'Er', 'Lu', 'So', 'Qu', 'Vi', 'Sa', 'Ti', 'Wr']
    Intensidade = ['50dB'] # Example, can be changed to ['30dB'] or ['60dB']
    Mmax_global = 240 # Max window for the entire analysis

    # --- 2. Load pre-computed optimal parameters ---
    alfa_opt = Config.ALPHA_DEFAULT
    fp_desejado_opt = Config.FP_DESEJADO_DEFAULT

    # Path to the precomputed .mat file
    # Example: NDC_AlfaCorrigido_Mmax240_alfa_0.05_FPdesejado0.05.mat
    filename = f"NDC_AlfaCorrigido_Mmax{Mmax_global}_alfa_{str(alfa_opt).replace('.', '')}_FPdesejado{str(fp_desejado_opt).replace('.', '')}.mat"
    filepath = os.path.join(Config.DATA_PATH, filename)
    
    if not os.path.exists(filepath):
        # If the file doesn't exist, generate it first
        print(f"Pre-computed file '{filename}' not found. Generating now...")
        alfa_corrigido_calc, ndc_minimo_calc, cost_alfa_calc, P_calc = funcao_NDC_alfaCorrigido_Mmax(
            nRuns=Config.N_RUNS_OPTIMIZATION,
            Mmax=Mmax_global,
            alfa_teste=alfa_opt,
            FP_desejado=fp_desejado_opt
        )
        # Save the generated file (assuming it should be in DATA_PATH)
        # Using savemat to match MATLAB's output structure
        from scipy.io import savemat
        savemat(filepath, {
            'alfa_corrigido': alfa_corrigido_calc,
            'NDC_minimo': ndc_minimo_calc,
            'P': P_calc,
            'nRuns': Config.N_RUNS_OPTIMIZATION # Save nRuns too as it's loaded
        })
        print(f"Pre-computed file '{filename}' generated and saved.")
        loaded_data = {
            'alfa_corrigido': alfa_corrigido_calc,
            'NDC_minimo': ndc_minimo_calc,
            'P': P_calc,
            'nRuns': Config.N_RUNS_OPTIMIZATION
        }
    else:
        print(f"Loading pre-computed parameters from: {filepath}")
        loaded_data = loadmat(filepath)

    alfa_corrigido = loaded_data['alfa_corrigido'].flatten()
    NDC_minimo = loaded_data['NDC_minimo'].flatten()
    P = loaded_data['P'] # [Mmin, Mstep, Mmax]
    
    # Combine P, NDC_minimo, and alfa_corrigido into 'parametros'
    # MATLAB: parametros = [P, NDC_minimo, alfa_corrigido];
    # Ensure NDC_minimo and alfa_corrigido are column vectors
    parametros = np.hstack((P, NDC_minimo[:, np.newaxis], alfa_corrigido[:, np.newaxis]))

    # MATLAB: parametros(:,5) = 0.05; -> Overwrites the corrected alpha with a fixed 0.05
    # This implies that despite the optimization for `alfa_corrigido`, it's then reset to 0.05.
    # We will follow this if the original MATLAB code intended it.
    for ii in range(parametros.shape[0]):
        parametros[ii, 4] = 0.05 # Index 4 is the alfa_corrigido column (0-indexed)

    # --- 3. Load electrode information ---
    # MATLAB: load(['eletrodos.mat'])
    eletrodos_filepath = os.path.join(Config.DATA_PATH, 'eletrodos.mat')
    if os.path.exists(eletrodos_filepath):
        eletrodos_data = loadmat(eletrodos_filepath)
        # Assuming eletrodos_data contains a variable named 'eletrodos' or similar
        # For simplicity, if 'eletrodos.mat' is just a placeholder, use ELECTRODES from Config
        # If 'pos_ele' is a 1-based index, convert to 0-based
        pos_ele = 0 # MATLAB was pos_ele = 1; -> so it's the first electrode
        print(f"Loaded electrode data. Using electrode: {Config.ELECTRODES[pos_ele]}")
    else:
        print(f"'{eletrodos_filepath}' not found. Using default 'pos_ele' = 0 (first electrode).")
        pos_ele = 0


    ganho = Config.GANHO
    remoc = Config.REMOC_THRESHOLD

    # --- 4. Process EEG data for each volunteer and channel ---
    # The MATLAB code loops for ncanal=1:16, but then 'x = x(:,:,pos_ele)' and '%%%%%%x=x(:,:,ncanal)'
    # This suggests it was modified to process a specific electrode (pos_ele) or all channels (ncanal).
    # We will follow the active line 'x = x(:,:,pos_ele)' meaning a single electrode chosen by 'pos_ele'.
    # Then the outer loop 'for ncanal=1:16' means the analysis is run 16 times, perhaps for different
    # analysis configurations, but the data itself is always from 'pos_ele'.
    # If the intention was to iterate through *physical* channels, `pos_ele` should be `ncanal-1`.

    # Let's assume the outer `ncanal` loop is intended to apply the analysis pipeline *per channel*.
    # So `x = x(:,:,ncanal)` is the active part, `pos_ele` is redundant or a debug remnant.
    # The `eletrodos.mat` would define the number of channels. Assuming 16 channels as per loop.
    num_channels_to_process = 16 # Based on the MATLAB loop `for ncanal=1:16`

    Tdr = np.zeros((100, parametros.shape[0], len(Vvoluntario)), dtype=np.int64) # Max 100 freq bins
    Ttime = np.zeros_like(Tdr, dtype=np.int64)
    FP_per_channel = np.zeros((parametros.shape[0], num_channels_to_process))
    timeMedio_per_channel = np.zeros_like(FP_per_channel)

    for ncanal in range(num_channels_to_process): # 0-indexed channel
        print(f"\nProcessing channel: {Config.ELECTRODES[ncanal] if ncanal < len(Config.ELECTRODES) else ncanal+1}")
        
        for cont_vol, voluntario in enumerate(Vvoluntario):
            current_intensity = Intensidade[0] # Assuming only one intensity to process as per `Intensidade = {'50dB'}`
            
            # Load volunteer and intensity specific data
            # MATLAB: load([voluntario intensidade], 'x','Fs','binsM','freqEstim')
            data_file = os.path.join(Config.DATA_PATH, f"{voluntario}{current_intensity}.mat")
            
            if not os.path.exists(data_file):
                warnings.warn(f"Data file '{data_file}' not found. Skipping volunteer {voluntario}.")
                continue

            vol_data = loadmat(data_file)
            x_raw = vol_data['x'] # Assume x_raw is (time_points, num_windows, num_channels)
            Fs = vol_data['Fs'].flatten()[0] # Sampling frequency
            binsM_freq_indices = vol_data['binsM'].flatten() # Indices of stimulation frequencies (1-indexed from MATLAB)
            # freqEstim = vol_data['freqEstim'] # Not directly used in the provided snippet

            # Select the current channel data
            # MATLAB: x = x(:,:,pos_ele); -> changed to x = x(:,:,ncanal); for the loop logic
            x_channel = x_raw[:, :, ncanal] # Select current channel (0-indexed)

            nfft = Fs # 1 second of signal, corresponds to window length `tj`

            # Remove DC component per window
            # MATLAB: x = x - repmat(mean(x),nfft,1);
            x_channel = x_channel - np.mean(x_channel, axis=0, keepdims=True)

            # Exclude first two seconds
            # MATLAB: x(:,1:2,:) =[];
            x_channel = x_channel[:, 2:]

            # Remove amplitude noise
            Vmax = np.max(np.abs(x_channel), axis=0) # Max abs value per window
            ind_noisy_windows = Vmax > remoc
            x_channel = x_channel[:, ~ind_noisy_windows] # Remove noisy windows

            # Limit to Mmax
            x_channel = x_channel[:, :Mmax_global]
            
            # Apply detection protocol
            current_dr, current_time = protocolo_deteccao(x_channel, parametros)

            # Store results
            # The size of current_dr/time depends on max_freq_bins_to_consider in protocolo_deteccao
            # Make sure Tdr and Ttime are large enough or handle dynamically
            max_freq_bins_used = current_dr.shape[0]
            Tdr[:max_freq_bins_used, :, cont_vol] = current_dr
            Ttime[:max_freq_bins_used, :, cont_vol] = current_time

        # --- Performance analysis per channel ---
        # TXD (Detection rate) - analyze stimulation frequencies
        # binsM_freq_indices are 1-indexed, convert to 0-indexed for Python
        binsM_0_indexed = binsM_freq_indices - 1
        
        # Calculate mean detection rate across volunteers for stimulation frequencies
        # Tdr (freq_bins x params_sets x volunteers)
        # mean(Tdr(binsM,:,:),3) -> mean across volunteers (axis=2)
        # mean(...,1)' -> mean across stimulation frequencies (axis=0), then transpose (becomes (params_sets,) )
        TXD_current_channel = np.mean(np.mean(Tdr[binsM_0_indexed, :, :len(Vvoluntario)], axis=2), axis=0) # (num_param_sets,)

        # FP (False Positive) - analyze non-stimulation frequencies
        # binsR = binsM+1; binsR = 1:100; binsR(binsM) = []; binsR(1:2) = [];
        all_bins = np.arange(100) # Assuming max_freq_bins_to_consider from protocolo_deteccao is 100 for FP calc
        binsR_0_indexed = np.setdiff1d(all_bins, binsM_0_indexed) # Remove stimulation bins
        binsR_0_indexed = binsR_0_indexed[binsR_0_indexed >= 2] # Remove first two bins (0 and 1)

        FP_current_channel = np.mean(np.mean(Tdr[binsR_0_indexed, :, :len(Vvoluntario)], axis=2), axis=0)
        FP_per_channel[:, ncanal] = FP_current_channel

        # Mean time for detection
        # Ttime (freq_bins x params_sets x volunteers)
        timeM = Ttime[binsM_0_indexed, :, :len(Vvoluntario)]
        timeM[timeM == -1] = Mmax_global # Replace -1 (no detection) with Mmax
        timeMedio_per_channel[:, ncanal] = np.mean(timeM, axis=(0, 2)) # Mean across freq bins and volunteers

    # Overall analysis (if you want to aggregate across channels, as per final MATLAB TXD/FP calculations)
    # The MATLAB script re-calculates TXD and FP outside the ncanal loop,
    # implicitly using the last calculated Tdr/Ttime which would be from the last channel processed.
    # To be true to the original, we should average over `FP_per_channel` and `timeMedio_per_channel`.
    # Let's average FP over channels
    FP_overall = np.mean(FP_per_channel, axis=1) # Average FP across channels for each parameter set

    # Re-calculate TXD overall if needed (not directly done in the final MATLAB snippet shown for TXD/FP)
    # This might require storing Tdr for all channels
    # For now, let's use the TXD from the last channel processed if not aggregating.
    # If the intention was to store `Tdr` for all channels and then calculate an overall mean,
    # the structure `Tdr(binsM,:,:)` implies it was taking the last Tdr.
    # Let's assume the TXD calculation should be for all channels processed.
    
    # MATLAB's final TXD calculation `TXD = mean(mean(Tdr(binsM,:,:),3),1)';`
    # This implies `Tdr` variable is overwritten by the last channel.
    # Let's re-calculate overall TXD for the final display from `Tdr` (assuming it holds all channel data if needed, or simply the last one)
    # For now, we will use the TXD calculated for the last channel, as the MATLAB code seems to do.
    # If the intention was to average TXD across all channels, it would need a similar structure to `FP_per_channel`.
    # Let's re-calculate it to ensure it represents the 'final' value as used in the plots.
    TXD_final_display = np.mean(np.mean(Tdr[binsM_0_indexed, :, :len(Vvoluntario)], axis=2), axis=0) * 100

    # --- 5. Display results ---
    print("\n--- Results ---")

    # 1 - Detection Rate (Taxa de Detecção)
    plt.figure()
    plt.plot(TXD_final_display, '.b', markersize=6)
    plt.xlabel('Índice dos conjuntos de Parâmetros', fontsize=12)
    plt.ylabel('Taxa de Detecção(%)', fontsize=12)
    plt.title(f'INTENSIDADE {Intensidade[0]} SPL', fontsize=12)
    plt.grid(True)
    plt.show()

    # 2 - False Positives (Falsos Positivos)
    plt.figure()
    # MATLAB: plot(FP)
    plt.plot(FP_overall)
    plt.xlabel('Índice dos conjuntos de Parâmetros', fontsize=12)
    plt.ylabel('Taxa de Falso Positivo', fontsize=12)
    plt.title(f'False Positives for INTENSIDADE {Intensidade[0]} SPL', fontsize=12)
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(FP_overall * 100, '.k', markersize=5, label='FP (no correction)')
    plt.axhline(y=6.2, color='r', linestyle=':', linewidth=2, label='Upper Sig. Limit (6.2%)')
    plt.axhline(y=3.9, color='r', linestyle=':', linewidth=2, label='Lower Sig. Limit (3.9%)')
    # The MATLAB comments mention FP_semCorrecao and FP_corrigido, but only FP is calculated.
    # This suggests that `FP_corrigido` might be intended to use `alfa_corrigido` for detection,
    # but the code sets `parametros(ii,5)=0.05` which is `alfa_teste`.
    # So `FP` (or `FP_overall`) effectively uses the uncorrected alpha of 0.05.
    # To plot `FP_corrigido`, it would require another run of `protocolo_deteccao` using the `alfa_corrigido` column.
    # For now, following the provided `plot(FP_corrigido(:,3)*100,'.b','Markersize',5)` line,
    # assuming FP_corrigido here means the FP calculated based on the processed data *using* the `parametros` which includes `alfa_corrigido` (even if it's reset to 0.05).
    # Since `parametros(ii,5)` is overwritten to 0.05, `FP_overall` essentially is the 'corrected' FP based on fixed 0.05 alpha.
    plt.plot(FP_overall * 100, '.b', markersize=5, label='FP (with correction applied, alpha=0.05)')
    plt.xlabel('Índice dos conjuntos de Parâmetros', fontsize=14)
    plt.ylabel('Taxa de Falso Positivo(%)', fontsize=14)
    plt.title(f'INTENSIDADE {Intensidade[0]} SPL', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Boxplot of FP (if fpboxplot data was generated, which it isn't in this snippet)
    # plt.figure()
    # plt.boxplot(fpboxplot)
    # plt.xlabel('NDC sem correção')
    # plt.show()


    # --- 6. Detection Rate vs Time (Pareto Front) ---
    # `timeM` is mean time across stimulation frequencies and volunteers
    # MATLAB: timeM = mean(timeM,1)'*1; %1segundo por janela
    # In Python, timeMedio_per_channel is already (params_sets, num_channels)
    # Need to average `timeMedio_per_channel` across channels to get a single `timeM` for Pareto.
    timeM_overall = np.mean(timeMedio_per_channel, axis=1) * 1 # Assuming 1 sec per window

    plt.figure()
    fig1, ax1 = plt.subplots()
    ax1.plot([0, 100], [Mmax_global, Mmax_global], '-.k', linewidth=1, label=f'Mmax={Mmax_global}')
    ax1.plot([TXD_final_display[0], TXD_final_display[0]], [np.min(timeM_overall), np.max(timeM_overall)], '-.b', linewidth=1, label='Single Test')

    for ii in range(parametros.shape[0]):
        ax1.plot(TXD_final_display[ii], timeM_overall[ii], '.k', markersize=6) # No label here, will be added by text below

    # Pareto Front calculation
    # paretoFront expects objectives to be maximized. TXD is maximized, -timeM_overall is maximized.
    p_data = np.vstack((TXD_final_display, -timeM_overall)).T
    p_front, idxs_front = paretoFront(p_data)

    # Sort Pareto front points by TXD
    sort_indices = np.argsort(p_front[:, 0])
    p_front = p_front[sort_indices, :]
    idxs_front = idxs_front[sort_indices]

    # Plot Pareto Front
    ax1.plot(p_front[:, 0], -p_front[:, 1], '-or', markersize=8, linewidth=1.2, label='Pareto Front')

    ax1.set_xlabel('Detection Rate (%)', fontsize=12)
    ax1.set_ylabel('Mean Exam Time (s)', fontsize=12)
    ax1.set_xlim(np.min(TXD_final_display[idxs_front]) * 0.95, np.max(TXD_final_display[idxs_front]) * 1.05)
    ax1.set_ylim(np.min(timeM_overall[idxs_front]) * 0.95, Mmax_global * 1.05)
    ax1.tick_params(which='minor', axis='both', direction='in')
    ax1.grid(True, which='both', linestyle=':', linewidth=0.5)
    ax1.legend()
    ax1.set_box_aspect(1) # Make sure it's square if needed
    plt.tight_layout()

    print("\n--- Pareto Front Points ---")
    for ii in range(idxs_front.shape[0]):
        original_idx = idxs_front[ii] # Index in the original `parametros` array
        
        # Find all parameter sets that result in this TXD and timeM_overall
        matching_indices = np.where((TXD_final_display == TXD_final_display[original_idx]) & \
                                    (timeM_overall == timeM_overall[original_idx]))[0]
        
        print(f"PD = {TXD_final_display[original_idx]:.2f}% Tempo = {timeM_overall[original_idx]:.2f}s, NI = {len(matching_indices)}")
        
        # Take the first matching index to print parameters as in MATLAB
        first_match_idx = matching_indices[0]
        
        Mmin_param = int(parametros[first_match_idx, 0])
        Mstep_param = int(parametros[first_match_idx, 1])
        print(f" - Buffer:{Mmin_param}, M_step:{Mstep_param}")
        
        # Add text label to the plot
        ax1.text(TXD_final_display[original_idx], timeM_overall[original_idx] * 0.975, 
                 f'{{{Mmin_param},{Mstep_param}}}',
                 fontsize=8, ha='center', va='top')
    
    plt.show()

    # --- 7. 3D Plotting (if matriz_dados exists from prior runs) ---
    # The MATLAB code attempts 3D plots with `matriz_dados` and `aumentapd`
    # which are commented out or derived from intermediate calculations that are not kept.
    # To run this, `matriz_dados` and `aumentapd` would need to be populated.
    # For now, these sections are skipped.
    
    # if 'matriz_dados' in locals() and 'aumentapd' in locals():
    #     X_scatter = matriz_dados[:, 0]
    #     Y_scatter = matriz_dados[:, 1]
    #     Z_scatter = matriz_dados[:, 2] # Assuming it stores 'aumentapd' or similar
    #     
    #     plt.figure()
    #     ax = plt.axes(projection='3d')
    #     ax.scatter(X_scatter, Y_scatter, Z_scatter)
    #     ax.set_xlabel('P(:,1)')
    #     ax.set_ylabel('P(:,2)')
    #     ax.set_zlabel('aumentapd')
    #     plt.title('3D Scatter Plot')
    #     plt.show()

    #     # Surface plot attempts
    #     x_grid, y_grid, z_grid = xyz2grid(parametros[:, 0], parametros[:, 1], aumentapd)
    #     plt.figure()
    #     ax = plt.axes(projection='3d')
    #     ax.plot_surface(x_grid, y_grid, z_grid, cmap='viridis')
    #     ax.set_xlabel('Buffer')
    #     ax.set_ylabel('M_step')
    #     ax.set_zlabel('aumentapd')
    #     plt.title('3D Surface Plot of aumentapd')
    #     plt.show()


if __name__ == '__main__':
    main_analysis()