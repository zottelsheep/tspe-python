from tspe.filters import TspeFilterPair, generate_filter_pairs

def test_generate_filter_pairs():
    a = [1]
    b = [1]
    c = [1]
    test_output = [
        TspeFilterPair(
            edge_filter=np.array([-1.0, 0.0, 2.0, 0.0, -1.0]),
            running_total_filter=np.array([1.0]),
            needed_padding=2,
            surrounding_window_size=1,
            observed_window_size=1,
            crossover_window_size=1,
        )
    ]

    function_output = generate_filter_pairs(a, b, c)

    for filter_pair_function, filter_pair_test in zip(function_output, test_output):
        assert np.array_equal(
            filter_pair_function.edge_filter, filter_pair_test.edge_filter
        )
        assert np.array_equal(
            filter_pair_function.running_total_filter,
            filter_pair_test.running_total_filter,
        )
        assert filter_pair_function.needed_padding == filter_pair_test.needed_padding
        assert (
            filter_pair_function.surrounding_window_size
            == filter_pair_test.surrounding_window_size
        )
        assert (
            filter_pair_function.observed_window_size
            == filter_pair_test.observed_window_size
        )
        assert (
            filter_pair_function.crossover_window_size
            == filter_pair_test.crossover_window_size
        )


