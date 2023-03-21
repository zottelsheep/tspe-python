from pathlib import Path
from typing import Optional, Union

import numpy as np

from elephant.conversion import BinnedSpikeTrain
from neo import SpikeTrain
from quantities import Quantity, millisecond
from scipy.io import loadmat

DEFAULT_BIN_SIZE = 1 * millisecond

def load_spike_train_example_mat(path: Union[Path,str],bin_size: Optional[Quantity] = None):

    if isinstance(path,str):
        path = Path(path)

    if not bin_size:
        bin_size = DEFAULT_BIN_SIZE


    data = loadmat(path,simplify_cells=True)['data']

    if 'asdf' not in data:
        raise ValueError('Incorrect Dataformat: Missing spiketrain_data in "asdf"')

    spiketrain_data = data['asdf']

    # Get number of electrodesa and recording_duration from last element of data array
    n_electrodes, recording_duration_ms = spiketrain_data[-1]

    # Create spiketrains
    spiketrains = []
    for spiketrain_raw in spiketrain_data[0:n_electrodes]:
        spiketrains.append(SpikeTrain(spiketrain_raw*millisecond, t_stop=recording_duration_ms*millisecond))

    spiketrains = BinnedSpikeTrain(spiketrains,bin_size=bin_size)

    return spiketrains

def load_spike_train_mat(path: Union[Path,str],bin_size: Optional[Quantity] = None):

    if isinstance(path,str):
        path = Path(path)

    if not bin_size:
        bin_size = DEFAULT_BIN_SIZE


    data = loadmat(path,simplify_cells=True)['SPIKEZ']

    if 'TS' not in data:
        raise ValueError('Incorrect dataformat: No timestamps in data under "TS"')

    spiketrain_timestamps = np.transpose(data['TS'])

    if 'PREF' not in data:
        raise ValueError('Incorrect dataformat: No metadata in data under "PREF"')

    spiketrain_metadata = data['PREF']

    # Get number of electrodesa and recording_duration from last element of data array
    n_electrodes = len(spiketrain_timestamps)
    recording_duration_ms = spiketrain_metadata['rec_dur'] * millisecond

    # Create spiketrains
    spiketrains = []
    for spiketrain_raw in spiketrain_timestamps[0:n_electrodes]:
        spiketrains.append(SpikeTrain(spiketrain_raw*millisecond, t_stop=recording_duration_ms))

    spiketrains = BinnedSpikeTrain(spiketrains,bin_size=bin_size)

    return spiketrains

