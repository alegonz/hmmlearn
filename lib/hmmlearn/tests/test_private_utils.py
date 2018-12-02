import numpy as np

from hmmlearn._utils import fill_covars


def test_fill_covars():
    full = np.arange(12).reshape(3, 2, 2) + 1
    np.testing.assert_equal(fill_covars(full, 'full', 3, 2), full)

    diag = np.arange(6).reshape(3, 2) + 1
    expected = np.array([[[1, 0], [0, 2]],
                         [[3, 0], [0, 4]],
                         [[5, 0], [0, 6]]])
    np.testing.assert_equal(fill_covars(diag, 'diag', 3, 2), expected)

    tied = np.arange(4).reshape(2, 2) + 1
    expected = np.array([[[1, 2], [3, 4]],
                         [[1, 2], [3, 4]],
                         [[1, 2], [3, 4]]])
    np.testing.assert_equal(fill_covars(tied, 'tied', 3, 2), expected)

    spherical = np.array([1, 2, 3])
    expected = np.array([[[1, 0], [0, 1]],
                         [[2, 0], [0, 2]],
                         [[3, 0], [0, 3]]])
    np.testing.assert_equal(fill_covars(spherical, 'spherical', 3, 2), expected)
