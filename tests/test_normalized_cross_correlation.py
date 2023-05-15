
def test_normalized_cross_correlation():
    # Generate Spiketrains
    delay_time = 5
    spike_times = [3, 4, 5] * ms
    spike_times_delayed = spike_times + delay_time * ms

    spiketrains = BinnedSpikeTrain(
        [
            SpikeTrain(spike_times, t_stop=20.0 * ms),
            SpikeTrain(spike_times_delayed, t_stop=20.0 * ms),
        ],
        bin_size=1 * ms,
    )

    test_output = np.array([[[0.0, 0.0], [1.1, 0.0]], [[0.0, 1.1], [0.0, 0.0]]])

    function_output = normalized_cross_correlation(
        spiketrains, [-delay_time, delay_time]
    )

    assert np.allclose(function_output, test_output, 0.1)
