from elephant.conversion import BinnedSpikeTrain
import numpy as np


def normalized_cross_correlation(spike_trains: BinnedSpikeTrain, delay_time:int = 0) -> np.ndarray:
    r"""normalized cross correlation using std deviation

    Computes the normalized_cross_correlation between all
    Spiketrains inside a BinnedSpikeTrain-Object at a given delay_time

    The underlying formula is:

    .. math::
        NCC_{X\arrY(d)} = \frac{1}{N_{bins}}\sum_{i=-\inf}^{\inf}{\frac{(y_{(i)} - \bar{y}) \cdot (x_{(i-d) - \bar{x})}{\sigma_x \cdot \sigma_y}}

    """

    n_bins = spike_trains.n_bins

    # Get sparse array of BinnedSpikeTrain
    # TODO: Check effekts of binary spike-train (spike_trains.binarize())
    spike_trains_array = spike_trains.to_sparse_array()

    # Get mean of spike_train
    ones = np.ones((n_bins,1))
    spike_trains_mean = ( spike_trains_array * ones ) / n_bins
    spike_trains_zeroed = spike_trains_array - spike_trains_mean

    # Get std deviation of spike trains
    spike_trains_std = np.std(spike_trains_zeroed,axis=1)

    # Compute cross correlation at given delay time
    # Uses theoretical zero-padding for shifted values,
    # but since $0 \cdot x = 0$ values can simply be omitted
    if delay_time >= 0:
        CC = spike_trains_zeroed[:,delay_time:] * np.transpose(spike_trains_zeroed[:,:-delay_time or None])
    else:
        CC = spike_trains_zeroed[:,:delay_time] * np.transpose(spike_trains_zeroed[:,-delay_time or None:])

    # Normalize using std deviation
    std_factors = spike_trains_std * np.transpose(spike_trains_std)
    NCC = CC / std_factors / n_bins

    return NCC

