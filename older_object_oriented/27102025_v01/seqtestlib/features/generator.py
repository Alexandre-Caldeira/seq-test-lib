# seqtestlib/features/generator.py
"""
Orchestrates the parallel generation of ORD features from raw simulated data.
This module reads configurations, sets up parallel tasks, executes them using
the high-performance kernels, and consolidates the results into a single HDF5 file.
"""

import os
import shutil
import ast
import h5py
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Dict, List, Any

from .. import config
from . import kernels

def _computation_worker(
    task: Dict[str, Any],
    input_file: str,
    output_folder: str,
    target_bins: np.ndarray
) -> None:
    """
    A single worker function for parallel processing. It computes metrics for one
    SNR/configuration pair and saves the result as a temporary .npy file.

    Args:
        task (Dict): A dictionary containing 'snr' and 'config' details for the job.
        input_file (str): Path to the source raw HDF5 data.
        output_folder (str): Directory to save the temporary result file.
        target_bins (np.ndarray): Array of frequency bin indices.
    """
    snr = task['snr']
    cfg = task['config']
    output_path = os.path.join(output_folder, f"result_{snr}_{cfg['name']}.npy")

    with h5py.File(input_file, 'r') as infile:
        target_ffts_dset = infile[snr]['target_ffts']
        power_spec_dset = infile[snr]['power_spectrum']
        num_trials = target_ffts_dset.shape[0]

        # Pre-allocate array for all results of this task
        final_results = np.zeros(target_ffts_dset.shape + (4,), dtype=np.float32)

        # Process in batches to manage memory
        for i in range(0, num_trials, config.TRIAL_BATCH_SIZE):
            end_idx = min(i + config.TRIAL_BATCH_SIZE, num_trials)
            
            metrics_batch = kernels.compute_ord_metrics_for_batch(
                target_ffts_dset[i:end_idx],
                power_spec_dset[i:end_idx],
                cfg['m_val'],
                cfg['l_val'],
                (cfg['mode'] == 'sliding'),
                target_bins
            )
            final_results[i:end_idx] = metrics_batch

    np.save(output_path, final_results)


class FeatureGenerator:
    """
    Generates and consolidates ORD features based on multiple configurations.
    """
    def __init__(self,
                 input_file: str = config.SIMULATED_RAW_DATA_FILE,
                 output_file: str = config.SIMULATED_FEATURES_FILE,
                 temp_folder: str = 'temp_feature_results',
                 cleanup_temp: bool = True):
        
        self.input_file = input_file
        self.output_file = output_file
        self.temp_folder = temp_folder
        self.cleanup_temp = cleanup_temp
        
        self.target_bins, self.snr_keys, self.analysis_configs = self._initialize_params()

    def _initialize_params(self) -> tuple:
        """Reads metadata from the input HDF5 file to prepare for processing."""
        with h5py.File(self.input_file, 'r') as f:
            fs = float(f.attrs['parameters_fs'])
            nfft = int(f.attrs['parameters_nfft'])
            all_freqs = ast.literal_eval(f.attrs['parameters_signal_freqs']) + \
                        ast.literal_eval(f.attrs['parameters_noise_freqs'])
            
            target_bins = np.array([int(freq * nfft / fs) for freq in all_freqs], dtype=np.int64)
            snr_keys = [k for k in f.keys() if k.startswith('snr_')]

        # Generate analysis configurations based on M_L_PAIRS from config
        analysis_configs = []
        for mode in ['sliding', 'fixed']:
            for m_val, l_val in config.M_L_PAIRS:
                analysis_configs.append({
                    'name': f'{mode}_m{m_val}_l{l_val}', 'mode': mode,
                    'm_val': m_val, 'l_val': l_val
                })

        return target_bins, snr_keys, analysis_configs

    def process_all(self):
        """
        Main method to run the entire feature generation workflow.
        """
        os.makedirs(self.temp_folder, exist_ok=True)
        
        tasks_to_compute, completed_tasks = self._setup_tasks()
        
        print(f"Found {len(completed_tasks)} tasks already in final HDF5 file.")
        print(f"Discovered {len(tasks_to_compute)} new tasks to compute.")

        if tasks_to_compute:
            print("\nStarting parallel feature computation...")
            Parallel(n_jobs=-1)(
                delayed(_computation_worker)(task, self.input_file, self.temp_folder, self.target_bins)
                for task in tqdm(tasks_to_compute, desc="Computing Tasks")
            )
        
        self._consolidate_results(completed_tasks)
        
        if self.cleanup_temp:
            print(f"\nCleaning up temporary folder: {self.temp_folder}")
            shutil.rmtree(self.temp_folder)
            
        print("\nFeature generation complete.")

    def _setup_tasks(self) -> tuple:
        """Determines which tasks need to be run by checking the output file."""
        tasks_to_compute = []
        completed_tasks = set()

        if os.path.exists(self.output_file):
            with h5py.File(self.output_file, 'r') as f:
                for snr_key in f.keys():
                    for cfg_key in f[snr_key].keys():
                        completed_tasks.add((snr_key, cfg_key))
        
        for snr in self.snr_keys:
            for cfg in self.analysis_configs:
                if (snr, cfg['name']) not in completed_tasks:
                    temp_path = os.path.join(self.temp_folder, f"result_{snr}_{cfg['name']}.npy")
                    if not os.path.exists(temp_path):
                        tasks_to_compute.append({'snr': snr, 'config': cfg})
                        
        return tasks_to_compute, completed_tasks

    def _consolidate_results(self, completed_tasks: set):
        """Merges temporary .npy files into the final HDF5 output."""
        temp_files = [f for f in os.listdir(self.temp_folder) if f.endswith('.npy')]
        if not temp_files:
            print("\nNo new results to consolidate.")
            return

        print(f"\nConsolidating {len(temp_files)} new results into: {self.output_file}")
        with h5py.File(self.output_file, 'a') as outfile:
            # Copy attributes from source if the output file is new
            if not outfile.attrs:
                with h5py.File(self.input_file, 'r') as infile:
                    for key, val in infile.attrs.items():
                        outfile.attrs[key] = val
            
            for temp_filename in tqdm(temp_files, desc="Consolidating"):
                base_name = temp_filename.replace('result_', '').replace('.npy', '')
                parts = base_name.split('_')
                
                snr_group_name = f"{parts[0]}_{parts[1]}"
                config_name = '_'.join(parts[2:])

                if (snr_group_name, config_name) not in completed_tasks:
                    snr_group = outfile.require_group(snr_group_name)
                    result_path = os.path.join(self.temp_folder, temp_filename)
                    metrics_data = np.load(result_path)
                    snr_group.create_dataset(config_name, data=metrics_data, compression="gzip")

if __name__ == '__main__':
    # This block allows running this script directly to generate features
    generator = FeatureGenerator()
    generator.process_all()