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
    max_delay: int = 25
):
    if not a:
        a = [3, 4, 5, 6, 7, 8]

    if not b:
        b = [2, 3, 4, 5, 6]

    if not c:
        c = [0]

    n_neurons, n_bins = spike_train_data.shape

    filter_pairs = generate_filter_pairs(a, b, c)

    # Calculate normalized cross corelation for different delays
    # The delay range ranges from 0 to max-delay and includes
    # padding for the filter convolution
    padding = max(a) + max(c)
    delay_range = list(range(-padding,max_delay + padding))
    NCC_d = normalized_cross_correlation(spike_train_data, delay_times=delay_range)

    # Apply edge and running total filter
    delay_matrix = np.zeros((n_neurons, n_neurons, max_delay-1))
    for filter in filter_pairs:
        # Select ncc_window based on needed filter padding
        NCC_window = NCC_d[:,:,padding-filter.needed_padding:max_delay+padding+filter.needed_padding]

        # Compute two convolutions with edge- and running total filter
        x1 = fftconvolve(NCC_window, np.expand_dims(filter.edge_filter,(0,1)), mode="valid",axes=2)
        x2 = fftconvolve(x1, np.expand_dims(filter.running_total_filter,(0,1)), mode="full",axes=2)

        delay_matrix += x2

    return delay_matrix

def generate_connectivity_matrix_from_delay_matrix(delay_matrix: np.ndarray) -> np.ndarray:

    # Take maxima of absolute of delays to get estimation for connectivity
    connectivity_matrix_index = np.argmax(np.abs(delay_matrix),axis=2,keepdims=True)
    connectivity_matrix = np.take_along_axis(delay_matrix,connectivity_matrix_index,axis=2).squeeze(axis=2)

    return connectivity_matrix


