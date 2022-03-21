from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

W = 64

def zero_center(a: np.ndarray) -> np.ndarray:
    return 2 * a - 1

def undo_zero_center(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + 1)

def normalise(a: np.ndarray) -> np.ndarray:
    return a / (W - 1)

def unnormalise(a: np.ndarray) -> np.ndarray:
    return a * (W - 1)