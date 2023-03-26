from matlab.engine import start_matlab
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from quantities import millisecond
import numpy as np

from tspe.io import load_spike_train_example_mat, load_spike_train_mat
from tspe.matlab_adapter import total_spiking_probability_edges_matlab
from tspe.tspe import (
    generate_connectivity_matrix_from_delay_matrix,
    total_spiking_probability_edges,
)

# Start Matlab
print('Starting MATLAB..')
eng = start_matlab()
print('Done!')

# Configure figure
fig = plt.figure()
fig.tight_layout()
fig.suptitle('TSPE – Implementation Comparisons')
gs = GridSpec(2,3,figure=fig)
sns.set_theme(style="white")
# cmap=sns.color_palette("Spectral", as_cmap=True)
cmap='icefire'

def plot_connectivity_matrix(index_y,index_x,matrix,title):
    ax = fig.add_subplot(gs[index_y,index_x])
    # ax.pcolormesh(matrix)
    sns.heatmap(matrix,ax=ax,cmap=cmap)
    ax.set_title(title)
    ax.set_aspect('equal')

# Load example spiketrain
print('Loading Spiketrain..')
example_mat = load_spike_train_example_mat('../../TSPE/ExampleSpikeTrain.mat', t_stop=1800000*millisecond)
print('Done!')

# example_mat = load_spike_train_mat('../reference/5 active electrodes.mat')

# Calculate python implementation
print('Calulating TSPE using Python implementation..')
delay_matrix = total_spiking_probability_edges(example_mat)
connectivity_matrix = generate_connectivity_matrix_from_delay_matrix(delay_matrix)
print('Done!')
plot_connectivity_matrix(0,0,connectivity_matrix,'Python implementation')

# Calculate MATLAB implementation
print('Calulating TSPE using MATLAB implementation..')
connectivity_matrix_matlab, _ = total_spiking_probability_edges_matlab(example_mat, matlab_engine=eng)
plot_connectivity_matrix(0,1,connectivity_matrix_matlab,'MATLAB implementation')
print('Done!')

# Average difference between implementations
print(f'Average difference: {np.nanmean(connectivity_matrix-connectivity_matrix_matlab)}')

# Plot differene
plot_connectivity_matrix(0,2,connectivity_matrix-connectivity_matrix_matlab,'Difference')

# Neglect self-connections
np.fill_diagonal(connectivity_matrix,0)
plot_connectivity_matrix(1,0,connectivity_matrix,'Python implementation – No self-connections')

np.fill_diagonal(connectivity_matrix_matlab,0)
plot_connectivity_matrix(1,1,connectivity_matrix_matlab,'MATLAB implementation – No self-connections')

# Plot difference with missing self-connections
plot_connectivity_matrix(1,2,connectivity_matrix-connectivity_matrix_matlab,'Difference – No self-connections')

print(f'Average difference – no self-connections: {np.nanmean(connectivity_matrix-connectivity_matrix_matlab)}')

plt.show()

