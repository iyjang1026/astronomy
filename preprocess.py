import numpy as np
import progressbar
import scipy
from multiprocessing import Process, Queue
import glob
import astropy.io.fits as fits
from astropy.stats import sigma_clip, sigma_clipped_stats
import sep
from scipy.ndimage import binary_dilation
from scipy.stats import mode
import os
import warnings

warnings.filterwarnings('ignore')

class Fits():
    def __init__(self):
        pass

    def file(path):
        return glob.glob(path + '/*.fits')
    def hdu(path):
        return fits.open(path)[0].data
    def hdr(path):
        return fits.open(path)[0].header
    
class Combine(Fits):
    def median_comb(array):
        median = np.median(array, axis=0)
        return median
    
    def nanmedian_comb(array):
        median = np.nanmedian(array, axis=0)
        return median

class Masking(Fits):
    def se_mask(arr):
        data = np.array(arr, dtype=float)
        data1 = data.astype(data.dtype.newbyteorder('='))

        bkg_data = sep.Background(data1)
        bkg = bkg_data.back()
        bkg_rms = bkg_data.rms()
        
        mean, median, std = sigma_clipped_stats(bkg_rms, cenfunc='median', stdfunc='mad_std', sigma=3)
        threshold = median - 3*std
        subd = data - bkg
        obj, seg_map = sep.extract(subd, threshold , segmentation_map=True)
        mask_map = np.array(seg_map)
        kernel = np.array([[1,1,1],[1,1,1],[1,1,1]]) #skimage.morphology.disk(3)
        mask_map_d = binary_dilation(mask_map, kernel, iterations=2)
        masked = np.where((mask_map_d!=0), np.nan, data)
        return masked

class Master(Combine):
    def master_bias(path):
        file = Fits.file(path)
        b_list = []
        for i in range(len(file)):
            hdu = Fits.hdu(file[i])
            b_list.append(hdu)
        master_bias = Combine.median_comb(np.array(b_list))
        return master_bias
    
    def master_dark(path, bias):
        dark_file = Fits.file(path)
        d_list = []
        for i in range(len(dark_file)):
            dark_data = Fits.hdu(dark_file[i])
            bias_subed = dark_data - bias
            d_list.append(bias_subed)
        master_dark = Combine.median_comb(np.array(d_list))
        return master_dark
    
    def dark_sky_flat(path, bias, dark):
        flat_file = Fits.file(path)
        scale_list = []
        mode_list = []
        for i in range(len(flat_file)):
            flat_data = Fits.hdu(flat_file[i])
            db_subed = flat_data - bias - dark
            masked = Masking.se_mask(db_subed)
            mode1 = mode(masked[~np.isnan(masked)])[0]
            scaled_data = (masked - mode1)/mode1
            scale_list.append(scaled_data)
            mode_list.append(mode1)
        mode_tot = np.array(mode_list)
        mean, median, std = sigma_clipped_stats(mode_tot, cenfunc='median', stdfunc='mad_std', sigma=3)
        mode0 = median
        flat_arr = np.array(scale_list)

        scaled_flat = Combine.nanmedian_comb(flat_arr)
        master_flat = scaled_flat * mode0 + mode0
        return master_flat
    

def ccdproc(path, bias, dark, flat, obj_name):
    os.mkdir(path + '/pp_obj')
    file = glob.glob(path + '/LIGHT/DB_subed/r/*.fits')
    for i in range(len(file)):
        n = format(i, '04')
        l_hdu = Fits.hdu(file[i])
        l_hdr = Fits.hdr(file[i])
        l_db = l_hdu - bias - dark
        mean, median, std = sigma_clipped_stats(flat, cenfunc='median', stdfunc='mad_std', sigma=3)
        l_final = l_db / (flat / median)
        fits.writeto(path + '/pp_obj/pp'+obj_name+str(n)+'.fits', l_final, header=l_hdr, overwrite=True)

def preproc(path):
    os.mkdir(path + '/process')
    bias = Master.master_bias(path + '/BIAS')
    fits.writeto(path + '/process/master_bias.fits', bias, overwrite=True)
    dark = Master.master_dark(path + '/DARK', bias)
    fits.writeto(path + '/process/master_dark.fits', dark, overwrite=True)
    flat = Master.dark_sky_flat(path + '/LIGHT/DB_subed/r', bias, dark)
    fits.writeto(path+'/process/master_flat.fits', flat, overwrite=True)
    ccdproc(path, bias, dark, flat, 'NGC5907')


preproc('/volumes/ssd/2025-05-25')