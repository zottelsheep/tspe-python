from time import perf_counter
from pathlib import Path
import json

from matlab.engine import start_matlab
from quantities import millisecond
import numpy as np
from scipy.io import loadmat

from tspe.io import load_spike_train_mat
from tspe.matlab_adapter import total_spiking_probability_edges_matlab
from tspe.tspe import tspe
from tspe.metrics import mean_squared_error, roc_curve

# Load Spiketrain
input_data = {
        "BA": Path('../reference/evaluation_data/BA/new_sim0_100.mat'),
        "CA": Path('../reference/evaluation_data/CA/new_sim0_100.mat'),
        "ER05": Path('../reference/evaluation_data/ER05/new_sim0_100.mat'),
        "ER10": Path('../reference/evaluation_data/ER10/new_sim0_100.mat'),
        "ER15": Path('../reference/evaluation_data/ER15/new_sim0_100.mat'),
        }
# RECORDING_DURATION = 1800000 * millisecond
RECORDING_DURATION = None
INPUT_FORMAT = 'SIMULATED'
OUTPUT_PATH = Path('./out')

print('==== Options ====')
OUTPUT_PATH.mkdir(parents=True,exist_ok=True)
print(f'Output-Path: {OUTPUT_PATH.resolve().absolute()}')


# Start Matlab
print('Starting MATLAB..')
eng = start_matlab()
print('Done!')

def benchmark_tspe(input_data):

    benchmark = {}

    print('  Loading Spiketrain..')
    spike_trains, original_data = load_spike_train_mat(input_data,
                                                       t_stop=RECORDING_DURATION,
                                                       format=INPUT_FORMAT
                                                       )
    print('  Done!')
    print('  Calulating TSPE using Python implementation..')

    t_start = perf_counter()
    connectivity_matrix, _ = tspe(spike_trains)
    np.fill_diagonal(connectivity_matrix,0)
    t_stop = perf_counter()
    benchmark["python"] = {
            "time": t_stop-t_start,
            "mean_squared_error": mean_squared_error(connectivity_matrix,original_data),
            "auc": roc_curve(connectivity_matrix,original_data)[3]
            }
    print(f'  Done! Took {benchmark["python"]["time"]:.2}s')

    # Calculate MATLAB implementation
    print('  Calulating TSPE using MATLAB implementation..')
    t_start = perf_counter()
    connectivity_matrix, _ = total_spiking_probability_edges_matlab(spike_trains, matlab_engine=eng)
    np.fill_diagonal(connectivity_matrix,0)
    t_stop = perf_counter()

    benchmark["matlab"] = {
            "time": t_stop-t_start,
            "mean_squared_error": mean_squared_error(connectivity_matrix,original_data),
            "auc": roc_curve(connectivity_matrix,original_data)[3]
            }

    print(f'  Done! Took {benchmark["matlab"]["time"]:.2}s')

    print('Done!')

    return benchmark


print('\n==== Starting Benchmark ====')

benchmark = {}

for label, input_file in input_data.items():
    print(f'Time tspe for {label}..')
    mat = loadmat(input_file,simplify_cells=True)['data']
    benchmark[label] = benchmark_tspe(input_file)
    metadata = {
            "NumEL_sim": mat["NumEL_sim"],
            "NumEL_rec": mat["NumEL_rec"],
            "recordingtime_ms": mat["recordingtime_ms"],
            "Simulator": mat["Simulator"],
            }
    benchmark[label].update(metadata)

print('\n==== Results ====')
print(json.dumps(benchmark,indent=2))

print('\nSaving to benchmark.json..')
with open(OUTPUT_PATH / 'benchmark.json','w') as f:
    json.dump(benchmark,f,indent=2)

print('Done!')

print('Output as Tables')

for label, data in benchmark.items():
    print(f'## {label}')
    print()
    print(f"| |Python|MATLAB")
    print(f"|-|-----|-----|")
    print(f"|Dauer|{data['python']['time']:.2f}|{data['matlab']['time']:.2f}|")
    print(f"|MSE|{data['python']['mean_squared_error']:.3f}|{data['matlab']['mean_squared_error']:.3f}|")
    print(f"|AUC|{data['python']['auc']:.3f}|{data['matlab']['auc']:.3f}|")

