from pathlib import Path

import pytest
import numpy as np

from tspe.tspe import tspe
from tspe.io import load_spike_train_mat
from tspe.metrics import roc_curve

TEST_DATA_DIR = Path(__file__).parent.parent.resolve() / "reference" / "evaluation_data"


@pytest.mark.parametrize(
    "datafiles",
    [
        TEST_DATA_DIR / "SW" / "new_sim0_100.mat",
    ],
)
def test_tspe_using_sim_data(datafiles):
    spiketrains, original_data = load_spike_train_mat(datafiles, format="SIMULATED")

    connectivity_matrix, _ = tspe(spiketrains)

    # Remove self-connections
    np.fill_diagonal(connectivity_matrix, 0)

    _, _, _, auc = roc_curve(connectivity_matrix, original_data)

    assert auc > 0.95
