from typing import List, Iterable, Union

from elephant.conversion import BinnedSpikeTrain
import numpy as np


def normalized_cross_correlation(
    spike_trains: BinnedSpikeTrain, delay_times: Union[int, List[int], Iterable[int]] = 0
) -> np.ndarray:
    r"""normalized cross correlation using std deviation

    Computes the normalized_cross_correlation between all
    Spiketrains inside a BinnedSpikeTrain-Object at a given delay_time

    The underlying formula is:

    .. math::
        NCC_{X\arrY(d)} = \frac{1}{N_{bins}}\sum_{i=-\inf}^{\inf}{\frac{(y_{(i)} - \bar{y}) \cdot (x_{(i-d) - \bar{x})}{\sigma_x \cdot \sigma_y}}}

    """

    n_neurons, n_bins = spike_trains.shape

    # Get sparse array of BinnedSpikeTrain
    # TODO: Check effekts of binary spike-train (spike_trains.binarize())
    spike_trains_array = spike_trains.to_sparse_array()

    # Get mean of spike_train
    ones = np.ones((n_bins, 1))
    spike_trains_mean = (spike_trains_array * ones) / n_bins
    spike_trains_zeroed = spike_trains_array - spike_trains_mean

    # Get std deviation of spike trains
    spike_trains_std = np.std(spike_trains_zeroed, axis=1)

    # Loop over delay times
    if isinstance(delay_times, int):
        delay_times = [delay_times]
    elif isinstance(delay_times, Iterable):
        delay_times = list(delay_times)

    NCC_d = np.zeros((len(delay_times), n_neurons, n_neurons))

    for delay_time in delay_times:
        # Uses theoretical zero-padding for shifted values,
        # but since $0 \cdot x = 0$ values can simply be omitted
        if delay_time >= 0:
            CC = spike_trains_array[:, delay_time:] * np.transpose(
                spike_trains_array[:, : -delay_time or None]
            )
        else:
            CC = spike_trains_array[:, :delay_time] * np.transpose(
                spike_trains_array[:, -delay_time or None :]
            )

        # Normalize using std deviation
        std_factors = spike_trains_std * np.transpose(spike_trains_std)
        NCC = CC / std_factors / n_bins

        # Compute cross correlation at given delay time
        NCC_d[delay_time, :, :] = NCC


    return NCC_d
