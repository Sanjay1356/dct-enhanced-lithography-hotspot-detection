import numpy as np


def zigzag_indices(n):
    indices = []

    for s in range(2 * n - 1):
        if s % 2 == 0:
            row_range = range(min(s, n - 1), max(-1, s - n), -1)
        else:
            row_range = range(max(0, s - n + 1), min(s, n - 1) + 1)

        for i in row_range:
            j = s - i
            if 0 <= j < n:
                indices.append((i, j))

    return np.array(indices, dtype=np.int32)


def keep_first_k(dct, k):
    n = dct.shape[0]

    if dct.shape[1] != n:
        raise ValueError(
            f"Expected square DCT matrix, got {dct.shape}"
        )

    indices = zigzag_indices(n)

    if k <= 0 or k > len(indices):
        raise ValueError(
            f"k must be between 1 and {len(indices)}, got {k}"
        )

    output = np.zeros_like(dct)

    selected = indices[:k]

    for i, j in selected:
        output[i, j] = dct[i, j]

    return output
