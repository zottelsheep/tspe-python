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

    max_length_edge_filter = 2 * max(a) + max(b) + 2 * max(c)

    # Calculate normalized cross corelation for different delays
    NCC_d = np.zeros((max_length_edge_filter + 25, n_neurons, n_neurons))
    for delay_time in range(max_length_edge_filter + 25):
        NCC_d[delay_time, :, :] = normalized_cross_correlation(
            spike_train_data, delay_time
        )


    # Apply edge and running total filter
    neuron_pairs = np.tril_indices(n_neurons)
    n_neurons_pairs = len(neuron_pairs[0])
    TSPE = np.zeros((n_neurons_pairs, max_length_edge_filter + 25))
    for edge_filter, running_total_filter in filter_pairs:
        for n_neuron_pair, (neuron_1, neuron_2) in enumerate(zip(*neuron_pairs)):
            x1 = np.convolve(NCC_d[:, neuron_1, neuron_2], edge_filter, 'valid')
            x2 = np.convolve( x1, running_total_filter, 'full')

            TSPE[n_neuron_pair, :len(x2)] += x2

    return TSPE
