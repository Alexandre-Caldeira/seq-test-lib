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

# def plot_simulation_results(results_df: pd.DataFrame, data_type: str = 'Simulated'):
#     """
#     Generates a series of performance plots (TPR, FPR, TTFD) for different alpha levels.

#     This function iterates through specified alpha values and creates faceted plots
#     comparing the performance of different detection methods across SNRs.

#     Args:
#         results_df (pd.DataFrame): The dataframe containing all simulation results.
#         data_type (str): The type of data being plotted (e.g., 'Simulated', 'Experimental').
#     """
#     if results_df.empty:
#         print(f"No results to plot for {data_type} data.")
#         return

#     # --- Data Preparation ---
#     results_df['Base Method'] = results_df['Feature'].apply(lambda x: 'CSM' if 'CSM' in x else 'MSC')
#     results_df['Type'] = results_df['Feature'].apply(lambda x: 'CumSum' if 'CumSum' in x else 'Standard')
#     results_df['Approach'] = results_df['Window'] + ', ' + results_df.get('Detection', 'ML')
#     results_df['Method'] = results_df['Base Method'] + ' ' + results_df['Type']
#     results_df['FPR Deviation'] = results_df['FPR'] - results_df['Alpha (%)'] / 100

#     # --- HOTFIX for TTFD=0 artifact in older data ---
#     # This specifically targets the 'Sliding, Per-Window' standard ORD case
#     # where the first window (index 0) could incorrectly register a detection.
#     mask = (
#         (results_df['Window'] == 'Sliding') &
#         (results_df.get('Detection') == 'Per-Window') &
#         (results_df['Type'] == 'Standard') &
#         (results_df['TTFD'] <= 3) # Use a small buffer to be safe
#     )
#     if mask.any():
#         print(f"Applying TTFD hotfix: Found and nullified {mask.sum()} potential artifact data points.")
#         results_df.loc[mask, 'TTFD'] = np.nan

#     # --- Plotting Configuration ---
#     plot_configs = [
#         {'metric': 'TPR', 'title_label': 'True Positive Rate (TPR)', 'y_label': 'TPR'},
#         {'metric': 'FPR Deviation', 'title_label': 'FPR Deviation', 'y_label': 'FPR Deviation (Actual - Target)'},
#         {'metric': 'TTFD', 'title_label': 'Time to First Detection', 'y_label': 'First Detection (Window Index)'}
#     ]
    
#     palette_lineplot = {'MSC': '#0072B2', 'CSM': '#009E73'}
#     markers, linestyles = {'Standard': 'o', 'CumSum': 'X'}, {'Standard': '', 'CumSum': (4, 2)}
    
#     # Apply a consistent journal style for all plots
#     style.set_journal_style(journal='nature', column_width='double')

#     print(f"\n--- Generating Plots for {data_type} Data ---")
#     for alpha_to_plot in config.SUMMARY_ALPHAS_FOR_PLOTS:
#         alpha_df = results_df[results_df['Alpha (%)'] == alpha_to_plot].copy()
#         if alpha_df.empty:
#             continue

#         print(f"\n--- Plots for Alpha = {alpha_to_plot}% ---")

#         for i, p_config in enumerate(plot_configs):
#             panel_char = chr(ord('a') + i)
#             interval_desc = f"({config.CI_PERCENT}% CI)" if p_config['metric'] != 'TTFD' else f"(Mean +/- {config.CI_PERCENT}% CI)"

#             g = sns.FacetGrid(alpha_df, col="Approach", col_wrap=2, height=5, aspect=1.4, sharey='row')
            
#             g.map_dataframe(
#                 sns.lineplot,
#                 x='SNR',
#                 y=p_config['metric'],
#                 hue='Base Method',
#                 style='Type',
#                 palette=palette_lineplot,
#                 markers=markers,
#                 dashes=linestyles,
#                 errorbar=('ci', config.CI_PERCENT)
#             )
            
#             g.add_legend(title='Method')
            
#             if p_config['metric'] == 'FPR Deviation':
#                 for ax in g.axes.flat:
#                     ax.axhline(0, ls='--', color='k', lw=1.5, zorder=0)
            
#             g.set_titles(col_template="{col_name}", size=12)
#             g.set_axis_labels("SNR (dB)", p_config['y_label'])
            
#             suptitle = (f'{panel_char}) {p_config["title_label"]} at {alpha_to_plot}% Alpha '
#                         f'{interval_desc}')
#             g.fig.suptitle(suptitle, y=1.03, fontsize=16, weight='bold')
            
#             plt.tight_layout(rect=[0, 0, 1, 0.97])
#             plt.show()

# seqtestlib/visualization/performance.py (Updated for Unified Plot)


def plot_simulation_results(results_df: pd.DataFrame):
    """
    Plots the performance of all models on a single TPR vs. FPR scatter plot.

    Args:
        results_df (pd.DataFrame): A DataFrame containing the combined results
                                   from all models, including the Q-Learning agent.
                                   It must contain 'TPR', 'FPR', and 'Feature' columns.
    """
    style.set_journal_style('nature', 'single')
    plt.figure(figsize=(10, 8))

    # Calculate the mean performance for each model/feature
    # The 'Feature' column now serves as the unique identifier for each model
    performance_summary = results_df.groupby('Feature')[['TPR', 'FPR']].mean().reset_index()

    # Get a colorblind-friendly color palette
    # Using seaborn's 'colorblind' palette is an excellent choice for accessibility
    num_models = len(performance_summary['Feature'].unique())
    palette = sns.color_palette("colorblind", n_colors=num_models)

    # Create the scatter plot
    sns.scatterplot(
        data=performance_summary,
        x='FPR',
        y='TPR',
        hue='Feature', # Use the 'Feature' column for color coding
        palette=palette,
        s=200,          # Increase marker size for better visibility
        style='Feature',# Use different markers for each model
        markers=True,
        edgecolor='black'
    )

    # --- Plot Customization ---
    plt.title('Performance Comparison of Detection Models', fontsize=16, pad=20)
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Set plot limits and add a reference line for random chance
    plt.xlim(0, max(0.2, performance_summary['FPR'].max() * 1.1)) # Adjust x-limit for clarity
    plt.ylim(0, 1.05)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Random Chance')

    # Improve legend
    plt.legend(title='Detection Method', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    plt.show()