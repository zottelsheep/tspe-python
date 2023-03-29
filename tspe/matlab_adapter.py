from typing import List, Optional
from pathlib import Path

import matlab.engine
from elephant.conversion import BinnedSpikeTrain
import numpy as np

from tspe.io import convert_to_sdf


def total_spiking_probability_edges_matlab(
    spike_train_data: BinnedSpikeTrain,
    a: Optional[List[int]] = None,
    b: Optional[List[int]] = None,
    c: Optional[List[int]] = None,
    matlab_engine: Optional[matlab.engine.MatlabEngine] = None,
    tspe_path: Optional[Path] = None,
):
    # Get tspe matlab function path
    if not tspe_path:
        tspe_path = Path(__file__).parent.parent / "reference"

    if not (tspe_path / "TSPE.m").exists():
        raise FileNotFoundError(tspe_path / "TSPE.m")

    # Start MatlabEngine if no engine was given
    if not matlab_engine:
        matlab_engine = matlab.engine.start_matlab()
        own_matlab_engine = True
    else:
        own_matlab_engine = False

    # Change path to tspe_path
    matlab_engine.cd(str(tspe_path.absolute()))

    # Convert spike_train_data into SpikeDataFormat
    sdf = convert_to_sdf(spike_train_data)

    # Call TSPE-Function via matlab
    connectivity_matrix, delay_matrix = matlab_engine.TSPE(sdf,nargout=2)

    # Handle return types
    connectivity_matrix = np.array(connectivity_matrix)
    delay_matrix = np.array(delay_matrix)

    # Quit matlab if own session was started
    if own_matlab_engine:
        matlab_engine.quit()

    return connectivity_matrix, delay_matrix
