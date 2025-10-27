# seqtestlib/features/kernels.py
"""
Contains the core, Numba-optimized functions for calculating ORD metrics.
These functions are designed for high performance on large batches of data.
"""

import numba
import numpy as np

@numba.jit(nopython=True, fastmath=True, cache=True)
def compute_ord_metrics_for_batch(
    target_ffts_batch: np.ndarray,
    power_spectrum_batch: np.ndarray,
    m_window_size: int,
    tfl_lobes: int,
    is_sliding_mode: bool,
    target_bins: np.ndarray,
) -> np.ndarray:
    """
    Computes MSC, CSM, TFL, and SNR_meas metrics for a batch of trials.

    This is the core computational engine, optimized with Numba for performance.
    It supports both sliding and fixed windowing modes.

    Args:
        target_ffts_batch (np.ndarray): Batch of complex FFTs for target channels.
                                        Shape: (num_trials, num_windows, num_freqs).
        power_spectrum_batch (np.ndarray): Batch of power spectra for target channels.
                                           Shape: (num_trials, num_windows, num_freqs).
        m_window_size (int): Size of the temporal window (M) for MSC/CSM calculation.
        tfl_lobes (int): Number of side lobes (L) for TFL calculation.
        is_sliding_mode (bool): If True, uses a sliding window; otherwise, fixed blocks.
        target_bins (np.ndarray): Array of frequency bin indices to analyze.

    Returns:
        np.ndarray: A matrix of calculated metrics.
                    Shape: (num_trials, num_windows, num_freqs, num_features).
                    Feature order: [MSC, CSM, TFL, SNR_meas].
    """
    num_trials, num_windows, num_freqs = target_ffts_batch.shape
    num_features = 4  # MSC, CSM, TFL, SNR_meas
    all_metrics = np.zeros((num_trials, num_windows, num_freqs, num_features), dtype=np.float32)

    for trial_idx in range(num_trials):
        for win_idx in range(num_windows):
            start_idx, end_idx, eff_win_size = 0, 0, 0

            # Determine the effective window for cumulative/block calculations
            if is_sliding_mode:
                num_windows_so_far = win_idx + 1
                eff_win_size = min(num_windows_so_far, m_window_size)
                if eff_win_size > 1:
                    start_idx = num_windows_so_far - eff_win_size
                    end_idx = num_windows_so_far
            else:  # Fixed mode
                if (win_idx + 1) % m_window_size == 0 and win_idx > 0:
                    eff_win_size = m_window_size
                    start_idx = (win_idx + 1) - eff_win_size
                    end_idx = win_idx + 1

            # Metrics based on the current window's power spectrum
            power_spec_current = power_spectrum_batch[trial_idx, win_idx, :]
            fft_mags_current = np.sqrt(power_spec_current)
            mean_mag = np.mean(fft_mags_current)
            std_mag = np.std(fft_mags_current)
            snr_meas = -10 * np.log10(mean_mag / std_mag) if (std_mag > 1e-9 and mean_mag > 1e-9) else 0.0

            for freq_idx in range(num_freqs):
                msc, csm = 0.0, 0.0
                if eff_win_size > 1:
                    # --- MSC Calculation ---
                    fft_window_slice = target_ffts_batch[trial_idx, start_idx:end_idx, :]
                    fft_freq_slice = fft_window_slice[:, freq_idx]
                    
                    num_msc = np.abs(np.sum(fft_freq_slice))**2
                    den_msc = eff_win_size * np.sum(np.abs(fft_freq_slice)**2)
                    msc = num_msc / den_msc if den_msc > 1e-9 else 0.0

                    # --- CSM Calculation ---
                    angles = np.angle(fft_freq_slice)
                    csm = np.mean(np.cos(angles))**2 + np.mean(np.sin(angles))**2

                # --- TFL Calculation ---
                target_bin = target_bins[freq_idx]
                power_signal = power_spec_current[target_bin]
                
                lobe_sum, lobe_count = 0.0, 0
                for i in range(target_bin - tfl_lobes, target_bin + tfl_lobes + 1):
                    if 0 <= i < num_freqs and i != target_bin:
                        lobe_sum += power_spec_current[i]
                        lobe_count += 1
                
                power_noise = lobe_sum / lobe_count if lobe_count > 0 else 0.0
                tfl = power_signal / power_noise if power_noise > 1e-9 else 0.0

                all_metrics[trial_idx, win_idx, freq_idx, :] = [msc, csm, tfl, snr_meas]
                
    return all_metrics


@numba.jit(nopython=True, fastmath=True, cache=True)
def compute_experimental_metrics(
    fft_all_windows: np.ndarray,
    fft_all_noise_ref: np.ndarray,
    m_window_size: int,
    tfl_lobes: int,
    is_sliding_mode: bool,
    target_bins: np.ndarray,
    nfft: int
) -> np.ndarray:
    """
    Calculates 7 features for experimental data, including TFG, Mag, and Phi.
    """
    n_clean_windows, _ = fft_all_windows.shape
    num_freqs = len(target_bins)
    num_features = 7  # msc, csm, tfg, tfl, snr_meas, mag_freq, phi_freq
    
    all_metrics = np.zeros((n_clean_windows, num_freqs, num_features), dtype=np.float32)
    power_spectrum_all = np.abs(fft_all_windows)**2

    for win_idx in range(n_clean_windows):
        start_idx, end_idx, eff_win_size = 0, 0, 0
        if is_sliding_mode:
            num_windows_so_far = win_idx + 1
            eff_win_size = min(num_windows_so_far, m_window_size)
            if eff_win_size > 1:
                start_idx = num_windows_so_far - eff_win_size
                end_idx = num_windows_so_far
        else:
            if (win_idx + 1) % m_window_size == 0 and win_idx > 0:
                eff_win_size = m_window_size
                start_idx = (win_idx + 1) - eff_win_size
                end_idx = win_idx + 1

        power_spec_current = power_spectrum_all[win_idx, :]
        fft_mags_current_half = np.abs(fft_all_windows[win_idx, :nfft // 2])
        std_mag = np.std(fft_mags_current_half)

        for freq_idx in range(num_freqs):
            target_bin = target_bins[freq_idx]
            
            msc, csm, tfg, mag_freq, phi_freq = 0.0, 0.0, 0.0, 0.0, 0.0
            if eff_win_size > 1:
                fft_slice_at_bin = fft_all_windows[start_idx:end_idx, target_bin]
                fft_noise_slice_at_bin = fft_all_noise_ref[start_idx:end_idx, target_bin]

                num_msc = np.abs(np.sum(fft_slice_at_bin))**2
                den_msc = eff_win_size * np.sum(np.abs(fft_slice_at_bin)**2)
                msc = num_msc / den_msc if den_msc > 1e-9 else 0.0
                
                angles = np.angle(fft_slice_at_bin)
                csm = np.mean(np.cos(angles))**2 + np.mean(np.sin(angles))**2
                
                power_signal_tfg = np.sum(np.abs(fft_slice_at_bin)**2)
                power_noise_tfg = np.sum(np.abs(fft_noise_slice_at_bin)**2)
                tfg = power_signal_tfg / (power_signal_tfg + power_noise_tfg) if (power_signal_tfg + power_noise_tfg) > 1e-9 else 0.0

                mag_freq = np.mean(np.abs(fft_slice_at_bin))
                phi_freq = np.angle(np.mean(fft_slice_at_bin))

            power_signal_tfl = power_spec_current[target_bin]
            lobe_sum, lobe_count = 0.0, 0
            for i in range(target_bin - tfl_lobes, target_bin + tfl_lobes + 1):
                if 0 <= i < nfft // 2 and i != target_bin:
                    lobe_sum += power_spec_current[i]
                    lobe_count += 1
            power_noise_mean = lobe_sum / lobe_count if lobe_count > 0 else 0.0
            tfl = power_signal_tfl / power_noise_mean if power_noise_mean > 1e-9 else 0.0

            power_at_bin = power_spec_current[target_bin]
            snr_meas = -10 * np.log10(power_at_bin / std_mag) if std_mag > 1e-9 and power_at_bin > 1e-9 else 0.0

            all_metrics[win_idx, freq_idx, :] = [msc, csm, tfg, tfl, snr_meas, mag_freq, phi_freq]
            
    return all_metrics