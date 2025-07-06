import numpy as np
import progressbar
import glob
import astropy.io.fits as fits
from astropy.stats import sigma_clipped_stats
import sep
from scipy.ndimage import binary_dilation
from scipy.stats import mode
import os
import warnings

warnings.filterwarnings('ignore')

class Fits:
    def __init__(self, path):
        self.path = glob.glob(path + '/*.fits')

class Data:
    def __init__(self, path):
        file = glob.glob(path + '/*.fits')
        self.hdu = fits.open(file)[0].data
        self.hdr = fits.open(file)[0].header

#print(glob.glob('/volumes/ssd/2025-06-27/BIAS' + '/*.fits'))
#print(Fits('/volumes/ssd/2025-06-27/BIAS/test').path)
#print(len(Fits('/volumes/ssd/2025-06-27/BIAS/test').path))
print(Data('/volumes/ssd/2025-06-27/BIAS/test').hdu)
#mport convert_fits

#print(convert_fits.convert_fits('/volumes/ssd/2025-06-27/BIAS').path)