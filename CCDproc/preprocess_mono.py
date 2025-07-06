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
import ray

warnings.filterwarnings('ignore')

class Fits:
    def __init__(self, path):
        self.path = glob.glob(path + '/*.fits')

    def mkdir(path, folder):
        if not os.path.exists(path + folder):
            os.mkdir(path + folder)
            

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

class Master(Fits):
    def master_bias(path):
        file = Fits(path + '/BIAS').path
        b_list = []
        for i in range(len(file)):
            hdu = fits.open(file[i])[0].data
            b_list.append(hdu)
        master_bias = Combine.median_comb(np.array(b_list))
        return master_bias
    
    def master_dark(path, bias):
        dark_file = Fits(path + '/DARK').path
        d_list = []
        for i in range(len(dark_file)):
            dark_data = fits.open(dark_file[i])[0].data
            bias_subed = dark_data - bias
            d_list.append(bias_subed)
        master_dark = Combine.median_comb(np.array(d_list))
        return master_dark
    
    def dark_sky_flat(path):
        flat_file = Fits(path + '/pp_obj').path
        scale_list = []
        mode_list = []
        bar0 = progressbar.ProgressBar(maxval=len(flat_file), widgets=['[',progressbar.Timer(),']',progressbar.Bar()]).start()
        for i in range(len(flat_file)):
            flat_data = fits.open(flat_file[i])[0].data
            db_subed = flat_data
            masked = Masking.se_mask(db_subed)
            mode1 = mode(masked[~np.isnan(masked)])[0]
            scaled_data = (masked - mode1)/mode1
            scale_list.append(scaled_data)
            mode_list.append(mode1)
            bar0.update(i)
        mode_tot = np.array(mode_list)
        mean, median, std = sigma_clipped_stats(mode_tot, cenfunc='median', stdfunc='mad_std', sigma=3)
        mode0 = median
        flat_arr = np.array(scale_list)
        bar0.finish()
        scaled_flat = np.array(Combine.nanmedian_comb(flat_arr))
        master_flat = scaled_flat * mode0 + mode0
        fits.writeto(path + '/process/master_flat.fits', master_flat, overwrite=True)
        #return master_flat

    def dome_flat(path):
        file = Fits(path + '/flat').path
        data = []
        for i in file:
            hdu = fits.open(i)[0].data 
            db_subed = hdu
            data.append(db_subed)
        flat_hdu = np.array(data)
        master_d_flat  = np.median(flat_hdu, axis=0)
        fits.writeto(path + '/process/master_flat.fits', master_d_flat, overwrite=True)
        
    
def db_sub(path, obj_name):
    bias = Master.master_bias(path)
    fits.writeto(path + '/process/master_bias.fits', bias, overwrite=True)
    dark = Master.master_dark(path, bias)
    fits.writeto(path + '/process/master_dark.fits', dark, overwrite=True)
    file = Fits(path + '/LIGHT').path
    Fits.mkdir(path, '/pp_obj')
    bar1 = progressbar.ProgressBar(maxval=len(file), widgets=['[',progressbar.Timer(),']',progressbar.Bar()]).start()
    for i in range(len(file)):
        n = format(i, '04')
        l_hdu = fits.open(file[i])[0].data
        l_hdr = fits.open(file[i])[0].header
        l_db = l_hdu - bias - dark
        fits.writeto(path + '/pp_obj/pp'+obj_name+str(n)+'.fits', l_db, header=l_hdr, overwrite=True)
        bar1.update(i)
    bar1.finish()

def flat_corr(path, obj_name):
    file = Fits(path).path
    flat = fits.open(path + '/process/master_flat.fits')[0].data
    Fits.mkdir(path, '/pp')
    bar1 = progressbar.ProgressBar(maxval=len(file), widgets=['[',progressbar.Timer(),']',progressbar.Bar()]).start()
    for i in range(len(file)):
        n = format(i, '04')
        l_hdu = fits.open(file[i])[0].data
        l_hdr = fits.open(file[i])[0].header
        l_db = l_hdu
        mean, median, std = sigma_clipped_stats(flat, cenfunc='median', stdfunc='mad_std', sigma=3)
        l_final = l_db / (flat / median)
        fits.writeto(path + '/pp/pp'+obj_name+str(n)+'.fits', l_final, header=l_hdr, overwrite=True)
        bar1.update(i)
    bar1.finish()

def astrometry(path, obj_name, ra, dec, radius):
    file = open(path+'/'+obj_name+'.sh', 'w')
    file.write(f'solve-field --index-dir /Users/jang-in-yeong/solve/index4100 --use-source-extractor -3 {ra} -4 {dec} -5 {radius} --no-plots *.fits \nrm -rf *.xyls *.axy *.corr *.match *.new *.rdls *.solved')
    file.close()

@ray.remote
def main(path, obj_name):
    Fits.mkdir(path, '/process')
    db_sub(path, obj_name)
    Master.dark_sky_flat(path)
    flat_corr(path, obj_name)
    
   

    
#ray.get(main.remote('/volumes/ssd/2025-06-27', 'Abell1656'))

