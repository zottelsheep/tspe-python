from typing import List, Optional

from elephant.conversion import BinnedSpikeTrain
import numpy as np
from scipy.signal import fftconvolve

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
    delay_matrix = np.zeros((n_neurons, n_neurons, max_delay))
    for edge_filter, running_total_filter in filter_pairs:
        x1 = fftconvolve(NCC_d, np.expand_dims(edge_filter,(0,1)), mode="valid",axes=2)
        x2 = fftconvolve(x1, np.expand_dims(running_total_filter,(0,1)), mode="full",axes=2)

        delay_matrix[:,:, : x2.shape[2] ] += x2

    print("Done!")

    return delay_matrix

def generate_connectivity_matrix_from_delay_matrix(delay_matrix: np.ndarray) -> np.ndarray:

    # Take maxima of absolute of delays to get estimation for connectivity
    connectivity_matrix = np.max(np.abs(delay_matrix),axis=2)

    return connectivity_matrix


