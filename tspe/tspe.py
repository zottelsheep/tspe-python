from typing import List, Optional

from elephant.conversion import BinnedSpikeTrain
import numpy as np

from tspe.filters import generate_filter_pairs
from tspe.normalized_cross_correlation import normalized_cross_correlation


def total_spiking_probability_edges(
    spike_train_data: BinnedSpikeTrain,
    a: Optional[List[int]] = None,
    b: Optional[List[int]] = None,
    c: Optional[List[int]] = None,
):
    if not a:
        a = [3, 4, 5, 6, 7, 8]

    if not b:
        b = [2, 3, 4, 5, 6]

    if not c:
        c = [0]

    n_neurons, n_bins = spike_train_data.shape

    filter_pairs = generate_filter_pairs(a, b, c)
    n_filter_pairs = len(filter_pairs)

    max_delay = 2 * max(a) + max(b) + 2 * max(c) + 25

    # Calculate normalized cross corelation for different delays
    NCC_d = normalized_cross_correlation(spike_train_data, delay_times=range(max_delay))


    # Apply edge and running total filter
    neuron_pairs = np.tril_indices(n_neurons)
    n_neurons_pairs = len(neuron_pairs[0])
    delay_matrix = np.zeros((n_neurons, n_neurons, max_delay))
    for edge_filter, running_total_filter in filter_pairs:
        for n_neuron_pair, (neuron_1, neuron_2) in enumerate(zip(*neuron_pairs)):
            # TODO: Test if fftconvole can replace forloop using axis argument
            x1 = np.convolve(NCC_d[neuron_1, neuron_2, :], edge_filter, "valid")
            x2 = np.convolve(x1, running_total_filter, "full")

            delay_matrix[neuron_1, neuron_2, : len(x2)] += x2

    # Copy upper triangular part of delay_matrix to bottom, since
    # it is the same information, due to NCC being symmetric
    # delay_matrix = delay_matrix + delay_matrix.swapaxes(0,1) - np.diag(np.diag(delay_matrix))

    return delay_matrix

def generate_connectivity_matrix_from_delay_matrix(delay_matrix: np.ndarray) -> np.ndarray:

    # Take maxima of absolute of delays to get estimation for connectivity
    connectivity_matrix = np.max(np.abs(delay_matrix),axis=2)

    return connectivity_matrix


