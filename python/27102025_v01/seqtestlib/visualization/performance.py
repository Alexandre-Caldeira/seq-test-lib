# seqtestlib/visualization/performance.py
"""
Functions for visualizing model performance metrics, such as TPR, FPR, and TTFD.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from . import style
from .. import config

def plot_simulation_results(results_df: pd.DataFrame, data_type: str = 'Simulated'):
    """
    Generates a series of performance plots (TPR, FPR, TTFD) for different alpha levels.

    This function iterates through specified alpha values and creates faceted plots
    comparing the performance of different detection methods across SNRs.

    Args:
        results_df (pd.DataFrame): The dataframe containing all simulation results.
        data_type (str): The type of data being plotted (e.g., 'Simulated', 'Experimental').
    """
    if results_df.empty:
        print(f"No results to plot for {data_type} data.")
        return

    # --- Data Preparation ---
    results_df['Base Method'] = results_df['Feature'].apply(lambda x: 'CSM' if 'CSM' in x else 'MSC')
    results_df['Type'] = results_df['Feature'].apply(lambda x: 'CumSum' if 'CumSum' in x else 'Standard')
    results_df['Approach'] = results_df['Window'] + ', ' + results_df.get('Detection', 'ML')
    results_df['Method'] = results_df['Base Method'] + ' ' + results_df['Type']
    results_df['FPR Deviation'] = results_df['FPR'] - results_df['Alpha (%)'] / 100

    # --- HOTFIX for TTFD=0 artifact in older data ---
    # This specifically targets the 'Sliding, Per-Window' standard ORD case
    # where the first window (index 0) could incorrectly register a detection.
    mask = (
        (results_df['Window'] == 'Sliding') &
        (results_df.get('Detection') == 'Per-Window') &
        (results_df['Type'] == 'Standard') &
        (results_df['TTFD'] <= 3) # Use a small buffer to be safe
    )
    if mask.any():
        print(f"Applying TTFD hotfix: Found and nullified {mask.sum()} potential artifact data points.")
        results_df.loc[mask, 'TTFD'] = np.nan

    # --- Plotting Configuration ---
    plot_configs = [
        {'metric': 'TPR', 'title_label': 'True Positive Rate (TPR)', 'y_label': 'TPR'},
        {'metric': 'FPR Deviation', 'title_label': 'FPR Deviation', 'y_label': 'FPR Deviation (Actual - Target)'},
        {'metric': 'TTFD', 'title_label': 'Time to First Detection', 'y_label': 'First Detection (Window Index)'}
    ]
    
    palette_lineplot = {'MSC': '#0072B2', 'CSM': '#009E73'}
    markers, linestyles = {'Standard': 'o', 'CumSum': 'X'}, {'Standard': '', 'CumSum': (4, 2)}
    
    # Apply a consistent journal style for all plots
    style.set_journal_style(journal='nature', column_width='double')

    print(f"\n--- Generating Plots for {data_type} Data ---")
    for alpha_to_plot in config.SUMMARY_ALPHAS_FOR_PLOTS:
        alpha_df = results_df[results_df['Alpha (%)'] == alpha_to_plot].copy()
        if alpha_df.empty:
            continue

        print(f"\n--- Plots for Alpha = {alpha_to_plot}% ---")

        for i, p_config in enumerate(plot_configs):
            panel_char = chr(ord('a') + i)
            interval_desc = f"({config.CI_PERCENT}% CI)" if p_config['metric'] != 'TTFD' else f"(Mean +/- {config.CI_PERCENT}% CI)"

            g = sns.FacetGrid(alpha_df, col="Approach", col_wrap=2, height=5, aspect=1.4, sharey='row')
            
            g.map_dataframe(
                sns.lineplot,
                x='SNR',
                y=p_config['metric'],
                hue='Base Method',
                style='Type',
                palette=palette_lineplot,
                markers=markers,
                dashes=linestyles,
                errorbar=('ci', config.CI_PERCENT)
            )
            
            g.add_legend(title='Method')
            
            if p_config['metric'] == 'FPR Deviation':
                for ax in g.axes.flat:
                    ax.axhline(0, ls='--', color='k', lw=1.5, zorder=0)
            
            g.set_titles(col_template="{col_name}", size=12)
            g.set_axis_labels("SNR (dB)", p_config['y_label'])
            
            suptitle = (f'{panel_char}) {p_config["title_label"]} at {alpha_to_plot}% Alpha '
                        f'{interval_desc}')
            g.fig.suptitle(suptitle, y=1.03, fontsize=16, weight='bold')
            
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.show()