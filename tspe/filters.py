import itertools
from typing import List, Tuple

import numpy as np

def generate_edge_filter(
    a: int,
    b: int,
    c: int,
) -> np.ndarray:
    r"""Generate an edge filter

    The edge filter is generated using following piecewise defined function:

    .. math::
        g_{(i)} = \begin{cases}
            - \frac{1}{a} & 0 \lt i \leq a \\
            \frac{2}{b} & a+c \lt i \leq a + b + c \\
            - \frac{1}{a} & a+b+2c \lt i \leq 2a + b + 2c
            \end{cases}

    """
    filter_length = (2 * a) + b + (2 * c) + 1
    i = np.linspace(0, filter_length, filter_length)

    conditions = [
        (i > 0) & (i <= a),
        (i > (a + c)) & (i <= a + b + c),
        (i > a + b + (2 * c)) & (i <= (2 * a) + b + (2 * c)),
    ]

    values = [-(1 / a), 2 / b, -(1 / a), 0]  # Default Value

    filter = np.piecewise(i, conditions, values)

    return filter


def generate_running_total_filter(b: int) -> np.ndarray:
    return np.ones(b)


def generate_filter_pairs(
    a: List[int],
    b: List[int],
    c: List[int],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generates filter pairs of edge and running total filter using all
    permutations of given parameters
    """
    filter_pairs = []

    for _a, _b, _c in itertools.product(a, b, c):
        g = generate_edge_filter(_a, _b, _c)
        h = generate_running_total_filter(_b)
        filter_pairs.append((g, h))

    return filter_pairs
