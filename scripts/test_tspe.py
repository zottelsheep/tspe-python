from time import perf_counter
from typing import Optional, Union, List
from pathlib import Path

from matlab.engine import start_matlab
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
from quantities import millisecond
import seaborn as sns

from tspe.io import load_spike_train_example_mat, load_spike_train_mat
from tspe.matlab_adapter import total_spiking_probability_edges_matlab
from tspe.tspe import (
    generate_connectivity_matrix_from_delay_matrix,
    total_spiking_probability_edges,
)

# ==== Options ====
OUTPUT_PATH = Path('./out')
COLORMAP = 'icefire'
SAVE_PLOTS = False
SHOW_PLOTS = True
PERFORM_NORM = False
DPI = 200

print('==== Options ====')
OUTPUT_PATH.mkdir(parents=True,exist_ok=True)
print(f'Output-Path: {OUTPUT_PATH.resolve().absolute()}')

print('\n==== Computation ====')

# Start Matlab
print('Starting MATLAB..')
eng = start_matlab()
print('Done!')


# Load example spiketrain
print('Loading Spiketrain..')
spike_trains, original_data = load_spike_train_example_mat('../reference/ExampleSpikeTrain.mat', t_stop=1800000*millisecond)

# spike_trains = load_spike_train_mat('../reference/5 active electrodes.mat')
# original_data = None

if original_data is not None:
    original_data = original_data.transpose()

print('Done!')


# ==== Calculating Connectivity Estimations ====

# Calculate python implementation
print('Calulating TSPE using Python implementation..')

t_start = perf_counter()
delay_matrix = total_spiking_probability_edges(spike_trains)
connectivity_matrix = generate_connectivity_matrix_from_delay_matrix(delay_matrix)
# Neglect self-connections
np.fill_diagonal(connectivity_matrix,0)
t_stop = perf_counter()
print(f'Done! Took {t_stop-t_start:.2}s')

# Calculate MATLAB implementation
print('Calulating TSPE using MATLAB implementation..')
t_start = perf_counter()
connectivity_matrix_matlab, _, _ = total_spiking_probability_edges_matlab(spike_trains, matlab_engine=eng)
# Neglect self-connections
np.fill_diagonal(connectivity_matrix_matlab,0)
t_stop = perf_counter()
print(f'Done! Took {t_stop-t_start:.2}s')

# ==== Metrics =====

print('\n==== Metrics ====')

def metrics(a,b,a_title,b_title):
    mean_squared_error = np.nanmean(np.power(a-b,2))
    print(f'MeanSquaredError [{a_title}-{b_title}]:\t {mean_squared_error}')
    return mean_squared_error


# Average difference between implementations
metrics(connectivity_matrix,connectivity_matrix_matlab,'Python','MATLAB')

if original_data is not None:

    # Metrics: python - orginal
    metrics(connectivity_matrix,original_data,'Python','Original')

    # Metrics: MATLAB - orginal
    metrics(connectivity_matrix_matlab,original_data,'MATLAB','Original')

# ==== Plotting ====

print('\n==== Plotting ====')

sns.set_theme(style="white")


# Helper function
def plot_connectivity_matrix(fig,gs,index_y,index_x,matrix,title):
    ax = fig.add_subplot(gs[index_y,index_x])
    sns.heatmap(matrix,ax=ax,cmap=COLORMAP, center=0)
    ax.set_title(title)
    ax.set_aspect('equal')

def comparison_plot(connectivity_matrix: np.ndarray,
                    connectivity_matrix_matlab: np.ndarray,
                    original_data: Optional[np.ndarray] = None,
                    annotations:Optional[Union[str,List[str]]]=None
                    ):

    # Title
    if isinstance(annotations,str):
        annotations = [annotations]

    if annotations:
        annotations_str = f"[{'-'.join(annotations)}]"
    else:
        annotations_str = ''

    title = 'TSPE – Implementation Comparisons'
    title = " ".join([title,annotations_str])

    print(f'Plotting "{title}"..')

    # Configure main figure
    fig = plt.figure(figsize=(18,5))
    fig.suptitle(title)
    gs = GridSpec(1,3,figure=fig)

    # Plot each connectivity_matrix
    plot_connectivity_matrix(fig,gs,0,0,connectivity_matrix,'Python Implementation')
    plot_connectivity_matrix(fig,gs,0,1,connectivity_matrix_matlab,'MATLAB Implementation')

    # Plot difference of connectivity_matrix
    plot_connectivity_matrix(fig,gs,0,2,np.abs(connectivity_matrix-connectivity_matrix_matlab),'Absolute Difference')

    if SAVE_PLOTS:
        save_path = OUTPUT_PATH / "".join(['tspe_implementation_comparison',annotations_str,'.png'])
        print(f'\tSaving to "{save_path}"..')
        plt.savefig(save_path,format='png',dpi=DPI)

    # If original plot with original side by side
    if original_data is not None:

        # Plot original data
        fig = plt.figure(figsize=(5,5))
        gs = GridSpec(1,1,figure=fig)

        plot_connectivity_matrix(fig,gs,0,0,original_data,'Original')

        if SAVE_PLOTS:
            save_path = OUTPUT_PATH / "".join(['connectivity_matrix_original',annotations_str,'.png'])
            print(f'\tSaving to "{save_path}"..')
            plt.savefig(save_path,format='png',dpi=DPI)

        # Same as above but with original_data included
        fig = plt.figure(figsize=(18,10))
        fig.suptitle(title)
        gs = GridSpec(2,3,figure=fig)

        # Plot each connectivity_matrix
        plot_connectivity_matrix(fig,gs,0,0,connectivity_matrix,'Python Implementation')
        plot_connectivity_matrix(fig,gs,0,1,connectivity_matrix_matlab,'MATLAB Implementation')

        # Plot difference of connectivity_matrix
        plot_connectivity_matrix(fig,gs,0,2,np.abs(connectivity_matrix-connectivity_matrix_matlab),'Absolute Difference')

        # Plot abs difference: python - orginal
        plot_connectivity_matrix(fig,gs,1,0,np.abs(connectivity_matrix-original_data),'Absolute Difference: Python - Original')

        # Plot abs difference: matlab - orginal
        plot_connectivity_matrix(fig,gs,1,1,np.abs(connectivity_matrix_matlab-original_data),'Absolute Difference: MATLAB - Original')

        # Plot original_data
        plot_connectivity_matrix(fig,gs,1,2,original_data,'Original Connections')

        if SAVE_PLOTS:
            save_path = OUTPUT_PATH / "".join(['tspe_implementation_comparison_difference_original',annotations_str,'.png'])
            print(f'\tSaving to "{save_path}"..')
            plt.savefig(save_path,format='png',dpi=DPI)

        # Plot Python, MATLAB, Original Comparison
        fig = plt.figure(figsize=(18,5))
        fig.suptitle(title)
        gs = GridSpec(1,3,figure=fig)

        # Plot each connectivity_matrix
        plot_connectivity_matrix(fig,gs,0,0,connectivity_matrix,'Python Implementation')
        plot_connectivity_matrix(fig,gs,0,1,connectivity_matrix_matlab,'MATLAB Implementation')
        plot_connectivity_matrix(fig,gs,0,2,original_data,'Original Connections')

        if SAVE_PLOTS:
            save_path = OUTPUT_PATH / "".join(['tspe_implementation_comparison_with_original',annotations_str,'.png'])
            print(f'\tSaving to "{save_path}"..')
            plt.savefig(save_path,format='png',dpi=DPI)

        # Plot individual comparisons

        # original with python
        fig = plt.figure(figsize=(10,5))
        gs = GridSpec(1,2,figure=fig)

        plot_connectivity_matrix(fig,gs,0,0,connectivity_matrix,'Python Implementation')
        plot_connectivity_matrix(fig,gs,0,1,original_data,'Original Connections')

        if SAVE_PLOTS:
            save_path = OUTPUT_PATH / "".join(['connectivity_matrix_tspe_python-original',annotations_str,'.png'])
            print(f'\tSaving to "{save_path}"..')
            plt.savefig(save_path,format='png',dpi=DPI)

        # original with matlab
        fig = plt.figure(figsize=(10,5))
        gs = GridSpec(1,2,figure=fig)

        plot_connectivity_matrix(fig,gs,0,0,connectivity_matrix_matlab,'MATLAB Implementation')
        plot_connectivity_matrix(fig,gs,0,1,original_data,'Original Connections')

        if SAVE_PLOTS:
            save_path = OUTPUT_PATH / "".join(['connectivity_matrix_tspe_matlab-original',annotations_str,'.png'])
            print(f'\tSaving to "{save_path}"..')
            plt.savefig(save_path,format='png',dpi=DPI)

comparison_plot(connectivity_matrix,
                connectivity_matrix_matlab,
                original_data)


# ==== Normalization ====

if PERFORM_NORM:

    print('\n==== Normalization ====')

    def norm(a):
        ratio = 2/(np.max(a)-np.min(a))
        # as you want your data to be between -1 and 1, everything should be scaled to 2,
        # if your desired min and max are other values, replace 2 with your_max - your_min
        shift = (np.max(a)+np.min(a))/2
        # now you need to shift the center to the middle, this is not the average of the values.
        return (a - shift)*ratio

    # Normalize all matricies
    connectivity_matrix_norm = norm(connectivity_matrix)
    connectivity_matrix_matlab_norm = norm(connectivity_matrix_matlab)
    if original_data is not None:
        original_data_norm = norm(original_data)
    else:
        original_data_norm = None

    metrics(connectivity_matrix_norm,connectivity_matrix_matlab_norm,'Python-Normalized','MATLAB-Normalized')

    comparison_plot(connectivity_matrix_norm,
                    connectivity_matrix_matlab_norm,
                    original_data_norm,
                    'Normalized',
                    )

# ==== Final ====

# Close matlab
eng.quit()

# Show Plots if specified
if SHOW_PLOTS:
    plt.show()
