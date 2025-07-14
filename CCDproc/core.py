import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import glob
class Hdu:
    def __init__(self, arr):
        self.arr = np.array(arr, dtype=np.float32)
