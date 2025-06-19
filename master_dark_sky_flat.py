import numpy as np
import astropy.io.fits as fits
from astropy.stats import sigma_clip
import matplotlib.pyplot as plt
import glob
import weakref
import progressbar
from scipy.stats import mode
import warnings
import sep
from scipy.ndimage import binary_dilation
import skimage

def se_mask(file):
    data0 = fits.open(file)[0].data
    #hdr = fits.open(file)[0].header
    data = np.array(data0, dtype=float)
    data1 = data.astype(data.dtype.newbyteorder('='))
    bkg_data = sep.Background(data1)
    bkg = bkg_data.back()
    bkg_rms = bkg_data.globalrms
    subd = data - bkg
    obj, seg_map = sep.extract(subd, 3*bkg_rms, segmentation_map=True)
    mask_map = np.array(seg_map)
    kernel = skimage.morphology.disk(3)
    mask_map_d = binary_dilation(mask_map, kernel, iterations=2)
    masked = np.where((mask_map_d!=0), np.nan, data0)
    return masked


def master_flat(path):
    file = glob.glob(path + '/N*.fits')
    data_tot = []
    bar0 = progressbar.ProgressBar(maxval=len(file), widgets=['[',progressbar.Timer(),']',progressbar.Bar()]).start()
    for t in range(7):
        masked_data = se_mask(file[t])
        data_tot.append(masked_data)
        bar0.update(t)
    bar0.finish()

    data_arr = np.array(data_tot)

    layer, x, y = data_arr.shape
    output = np.zeros((x,y))

    bar = progressbar.ProgressBar(maxval=x,widgets=['[',progressbar.Timer(),']', progressbar.Bar()]).start()

    warnings.filterwarnings('ignore')
    
    for i in range(x):
        for j in range(y):
            pixel_data = data_arr[:,i,j]
            weak_pixel = weakref.ref(pixel_data)()
            #pixel = sigma_clip(weak_pixel[~np.isnan(weak_pixel)], sigma_lower=6, sigma_upper=3, cenfunc='median', stdfunc='mad_std')
            mode0 = mode(weak_pixel[~np.isnan(weak_pixel)])
            output[i,j] += mode0[0]
        bar.update(i)
    bar.finish()
    fits.writeto(path+'/test_flat.fits', output, overwrite=True)

master_flat('/volumes/ssd/2025-06-16/LIGHT/DB_subed/r')
