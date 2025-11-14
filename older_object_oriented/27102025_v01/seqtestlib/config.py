# seqtestlib/config.py
"""
Global configuration file for the seq-test-lib library.
Centralizes paths, constants, and parameters for analysis, modeling, and visualization.
"""

import os

# --- 1. CORE PATHS ---
# Base directory for the project. Assumes this file is in a 'seqtestlib' subfolder.
# BASE_PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BASE_PROJECT_PATH = r"C:/PPGEE/Assessing CGST on ASSR/clean_code/assr-ord/history/260920205_improvingICASSP/"
# Input data paths
SIMULATED_RAW_DATA_FILE = os.path.join(BASE_PROJECT_PATH, 'data', 'train_v5.hdf5')
SIMULATED_FEATURES_FILE = os.path.join(BASE_PROJECT_PATH, 'data', 'results_v5_4_correct_structure.hdf5')
EXPERIMENTAL_RAW_DATA_DIR = 'C:/Users/alexa/experimental_data/todos/Sinais_EEG/'

# Output data paths
EXPERIMENTAL_FEATURES_FILE = os.path.join(BASE_PROJECT_PATH, 'data', 'experimental_features_highFreqNoise_filtered.hdf5')


# --- 2. GLOBAL ANALYSIS CONTROLS ---
RANDOM_STATE = 42
ALPHA_LEVELS = [10.0, 5.0, 1.0]  # Significance levels for statistical tests
CI_PERCENT = 95                 # Confidence interval for plots


# --- 3. DATA PREPROCESSING & FEATURE ENGINEERING PARAMETERS ---
# Frequencies of interest (Hz)
SIGNAL_FREQS = [82, 84, 86, 88, 90, 92, 94, 96]
NOISE_FREQS = [val * 2 for val in [131, 137, 149, 151, 157, 163, 167, 173]]

# Experimental data specifics
ELECTRODE_TO_USE = 0
ARTIFACT_THRESHOLD = 0.1 / 200

# Bandpass filter settings for experimental data
FILTER_ORDER = 4
LOW_CUT = 70.0
HIGH_CUT = 110.0

# Configurations for feature generation jobs (M = window size, L = lobes for TFL)
M_L_PAIRS = [
    (12, 8), (18, 10), (24, 5), (32, 7),
    (40, 9), (6, 4), (120, 2)
]


# --- 4. MACHINE LEARNING & SEQUENTIAL TEST PARAMETERS ---
ML_WINDOW_SIZE = 40  # M-value specifically for ML models (DANN, LGBM)

# DANN settings
DANN_EPOCHS = 10

# LightGBM settings
LGBM_N_ESTIMATORS = 100
LGBM_PARAMS = {
    'objective': 'binary', 'metric': 'auc', 'n_estimators': LGBM_N_ESTIMATORS,
    'learning_rate': 0.05, 'num_leaves': 31, 'verbose': -1,
    'random_state': RANDOM_STATE, 'n_jobs': -1
}


# --- 5. PERFORMANCE & SAMPLING CONTROLS ---
# For feature generation
TRIAL_BATCH_SIZE = 500

# For cross-validation and model training/testing
N_SPLITS_SIM = 3
TEST_SIZE_SIM = 0.3
N_SPLITS_ORD = 25
MIN_TEST_SAMPLES_ORD = 5

# To manage memory and training time
MAX_SIM_TRIALS_FOR_ML_TRAINING = 50    # Limit trials used for sourcing ML training data
MAX_TOTAL_TRAIN_SAMPLES_ML = 500      # Cap total windows for ML training set
MAX_TRAIN_SAMPLES_PER_SNR_ML = 300    # Cap windows per SNR for ML training
MAX_TEST_SAMPLES_PER_SNR_ML = 500     # Cap windows per SNR for ML evaluation


# --- 6. VISUALIZATION CONTROLS ---
SUMMARY_ALPHAS_FOR_PLOTS = [5.0, 1.0] # Alphas to generate summary plots for
BASE_METHOD_FOR_PLOTS = 'MSC'         # Base feature for ML model plots (e.g., DANN-MSC)