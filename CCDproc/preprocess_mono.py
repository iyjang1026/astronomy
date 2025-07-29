import numpy as np
import progressbar
import glob
import astropy.io.fits as fits
from astropy.stats import sigma_clipped_stats, sigma_clip
from scipy.stats import mode
import os
import warnings
import weakref
from mask import region_mask
import sys
import ray
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

class Fits:
    def __init__(self, path):
        self.path = glob.glob(path + '/*.fit')

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
    
class Master(Fits):
    def master_bias(path):
        file = Fits(path + '/BIAS').path
        b_list = []
        for i in range(len(file)):
            hdul = fits.open(file[i])
            hdu = hdul[0].data
            b_list.append(weakref.ref(hdu)())
        master_bias = Combine.median_comb(np.array(b_list))
        return master_bias.astype(np.float32)
    
    def master_dark(path, bias):
        dark_file = Fits(path + '/DARK').path
        d_list = []
        for i in range(len(dark_file)):
            hdul = fits.open(dark_file[i])
            dark_data = hdul[0].data
            bias_subed = dark_data - bias
            bias_subed.astype(np.float32)
            d_list.append(weakref.ref(bias_subed)())
            hdul.close()
        master_dark = Combine.median_comb(np.array(d_list))
        return master_dark.astype(np.float32)
    
    def masking(path):
        file = glob.glob(path +'/pp_obj/*.fit')
        Fits.mkdir(path, '/mask')
        bar1 = progressbar.ProgressBar(maxval=len(file), widgets=['[',progressbar.Timer(),']',progressbar.Bar()]).start()
        @ray.remote
        def mask(file, i):
            hdu = fits.open(file[i])[0].data 
            n = format(i, '04')
            mask = region_mask(hdu,1.5, 0.8)
            fits.writeto(path+'/mask/mask'+str(n)+'.fits', mask, overwrite=True)
            bar1.update(i)
        ray.get([mask.remote(file, i) for i in range(len(file))])
        bar1.finish()
        ray.shutdown()
        """
        for i in range(len(file)):
            hdu = fits.open(file[i])[0].data 
            n = format(i, '04')
            mask = region_mask(hdu,1.5)
            fits.writeto(path+'/mask/mask'+str(n)+'.fits', mask, overwrite=True)
            bar1.update(i)
        bar1.finish()
        """    
    
    def dark_sky_flat(path):
        flat_file = glob.glob(path +'/pp_obj/pp*.fit')
        mask_file = glob.glob(path + '/mask/*.fits')
        scale_list = []
        mode_list = []
        bar0 = progressbar.ProgressBar(maxval=len(flat_file), widgets=['[',progressbar.Timer(),']',progressbar.Bar()]).start()
        for i in range(len(flat_file)):
            flat_data, scaled_data, mode1, masked = None,None,None,None
            hdul = fits.open(flat_file[i])
            flat_data = weakref.ref(hdul[0].data)()
            mask = fits.open(mask_file[i])[0].data
            masked = np.where(mask==1,np.nan,flat_data)
            mode1 = mode(masked[~np.isnan(masked)])[0]
            scaled_data = np.array((masked - mode1)/mode1)
            scale_list.append(scaled_data.astype(np.float32))
            mode_list.append(mode1)
            hdul.close()
            bar0.update(i)
        mode_tot = np.array(mode_list)
        mean, mode0, std = sigma_clipped_stats(mode_tot, cenfunc='median', stdfunc='mad_std', sigma=3)
        flat_arr = np.array(scale_list).astype(np.float32)
        bar0.finish()
        scaled_flat = np.nanmedian(weakref.ref(flat_arr)(), axis=0)
        scaled_flat.astype(np.float32)
        master_flat = np.array((scaled_flat * mode0) + mode0, dtype=np.float32)
        fits.writeto(path + '/process/master_flat.fits', weakref.ref(master_flat)(), overwrite=True)
        bar0.finish()
        print(f'master flat has been made')

    def dome_flat(path):
        file = glob.glob(path +'/pp_obj/pp*.fit')
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
        hdul = fits.open(file[i])
        l_hdu = hdul[0].data
        l_hdr = hdul[0].header
        l_hdu.astype(np.float32)
        l_db = l_hdu - bias - dark
        l_db.astype(np.float32)
        fits.writeto(path + '/pp_obj/pp'+obj_name+'_'+str(n)+'.fit', weakref.ref(l_db)(), header=l_hdr, overwrite=True)
        hdul.close()
        bar1.update(i)
    bar1.finish()
    print(f'dark and bias are subed')

def flat_corr(path, obj_name):
    file = Fits(path + '/pp_obj').path
    flat = fits.open(path + '/process/master_flat.fits')[0].data
    Fits.mkdir(path, '/pp')
    bar1 = progressbar.ProgressBar(maxval=len(file), widgets=['[',progressbar.Timer(),']',progressbar.Bar()]).start()
    for i in range(len(file)):
        n = format(i, '04')
        l_hdu = fits.open(file[i])[0].data
        l_hdr = fits.open(file[i])[0].header
        l_db = l_hdu.astype(np.float32)
        mean, median, std = sigma_clipped_stats(flat, cenfunc='median', stdfunc='mad_std', sigma=3)
        l_final = l_db / (flat / median)
        l_final.astype(np.float32)
        fits.writeto(path + '/pp/pp'+obj_name+'_'+str(n)+'.fits', weakref.ref(l_final)(), header=l_hdr, overwrite=True)
        bar1.update(i)
    bar1.finish()
    print(f'preprocessing complete')

def astrometry(path, obj_name, ra, dec, radius):
    file = open(path+'/'+obj_name+'.sh', 'w')
    file.write(f'solve-field --index-dir /Users/jang-in-yeong/solve/index4100 --use-source-extractor -3 {ra} -4 {dec} -5 {radius} --no-plots *.fits \nrm -rf *.xyls *.axy *.corr *.match *.new *.rdls *.solved')
    file.close()

from sky_sub import sky_sub
def sky_subd(path, obj_name):
    if not os.path.exists(path + '/sky_subed2'):
        os.mkdir(path + '/sky_subed2')
    p = glob.glob(path + '/pp/pp*.fits')
    #m = glob.glob(path + '/mask/*.fits')
    bar1 = progressbar.ProgressBar(maxval=len(p), widgets=['[',progressbar.Timer(),']',progressbar.Bar()]).start()
    @ray.remote
    def sky(p,i):
        n = format(i, '04')
        hdu = fits.open(p[i])[0].data 
        hdr = fits.open(p[i])[0].header
        #mask = fits.open(m[i])[0].data
        subed, hdr = sky_sub(hdu,hdr)
        fits.writeto(path +'/sky_subed2/pp' + obj_name + str(n)+'.fits',subed , header=hdr, overwrite=True)
        bar1.update(i)
    ray.get([sky.remote(p,i) for i in range(len(p))])
    bar1.finish()
    ray.shutdown()

import time
def process(path, obj_name):
    start_time = time.time()
    Fits.mkdir(path, '/process')
    db_sub(path, obj_name)
    Master.masking(path)
    Master.dark_sky_flat(path)
    flat_corr(path, obj_name)
    end_time = time.time()
    eta = end_time - start_time
    print(f'{eta//60} min {eta-(eta//60)*60} seconds')



def full_proc(path, obj_name):
    process(path, obj_name)
    sky_subd(path, obj_name)

if __name__ == '__main__':
    #process('/volumes/ssd/intern/25_summer/M101_L', 'M101')
    full_proc('/volumes/ssd/intern/25_summer/M101_L', 'M101')
    #astrometry('/volumes/ssd/intern/25_summer/M101_L/sky_subed','M101','14:03:12.6','+54:20:55.5','1.5')

