import numpy as np
import progressbar
import glob
import astropy.io.fits as fits
from astropy.stats import sigma_clipped_stats, SigmaClip
import sep
from scipy.ndimage import binary_dilation
from scipy.stats import mode
import os
import warnings
from photutils.segmentation import detect_threshold
from mask import region_mask

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
        
        sigma = SigmaClip(sigma=3.0, maxiters=10)
        
        threshold = np.median(detect_threshold(data1, nsigma=3.0, sigma_clip=sigma))
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
        return master_bias.astype(np.float32)
    
    def master_dark(path, bias):
        dark_file = Fits(path + '/DARK').path
        d_list = []
        for i in range(len(dark_file)):
            dark_data = fits.open(dark_file[i])[0].data
            bias_subed = dark_data - bias
            d_list.append(bias_subed)
        master_dark = Combine.median_comb(np.array(d_list))
        return master_dark.astype(np.float32)
    
    def masking(path, color):
        import ray
        file = glob.glob(path +'/'+color+'/*.fits')
        Fits.mkdir(path+'/'+color, '/mask')
        bar1 = progressbar.ProgressBar(maxval=len(file), widgets=['[',progressbar.Timer(),']',progressbar.Bar()]).start()
        @ray.remote
        def mask(file, i):
            hdu = fits.open(file[i])[0].data 
            n = format(i, '04')
            mask = region_mask(hdu,1.5)
            fits.writeto(path+'/'+color+'/mask/mask'+str(n)+'.fits', mask, overwrite=True)
            bar1.update(i)
        ray.get([mask.remote(file, i) for i in range(len(file))])
        bar1.finish()
        ray.shutdown()

    def dark_sky_flat(path, color):
        flat_file = glob.glob(path + '/'+color+'/*.fits')
        mask_file = glob.glob(path+'/'+color+'/mask/*.fits')
        scale_list = []
        mode_list = []
        bar0 = progressbar.ProgressBar(maxval=len(flat_file), widgets=['[',progressbar.Timer(),']',progressbar.Bar()]).start()
        for i in range(len(flat_file)):
            flat_data = fits.open(flat_file[i])[0].data
            mask = fits.open(mask_file[i])[0].data 
            db_subed = flat_data.astype(np.float32)
            masked = np.where(mask!=0, np.nan, flat_data)
            mode1 = mode(masked[~np.isnan(masked)])[0]
            scaled_data = (masked - mode1)/mode1
            scale_list.append(scaled_data.astype(np.float32))
            mode_list.append(mode1)
            bar0.update(i)
        mode_tot = np.array(mode_list)
        mean, median, std = sigma_clipped_stats(mode_tot, cenfunc='median', stdfunc='mad_std', sigma=3)
        mode0 = median
        flat_arr = np.array(scale_list, dtype=np.float32)
        bar0.finish()
        scaled_flat = np.array(Combine.nanmedian_comb(flat_arr), dtype=np.float32)
        master_flat = scaled_flat * mode0 + mode0
        fits.writeto(path + '/process/master_flat_'+color+'.fits', master_flat.astype(np.float32), overwrite=True)
        #return master_flat

    def dome_flat(path, color):
        file = Fits(path + '/color_flat/'+color).path
        data = []
        for i in file:
            hdu = fits.open(i)[0].data 
            db_subed = hdu
            data.append(db_subed)
        flat_hdu = np.array(data)
        master_d_flat  = np.median(flat_hdu, axis=0)
        fits.writeto(path + '/process/master_d_flat_'+color+'.fits', master_d_flat, overwrite=True)
        
    
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

def flat_corr(path, obj_name, color):
    file = Fits(path + '/'+color).path
    flat = fits.open(path + '/process/master_flat_'+color+'.fits')[0].data
    Fits.mkdir(path, '/'+color+'_pp')
    bar1 = progressbar.ProgressBar(maxval=len(file), widgets=['[',progressbar.Timer(),']',progressbar.Bar()]).start()
    for i in range(len(file)):
        n = format(i, '04')
        l_hdu = fits.open(file[i])[0].data
        l_hdr = fits.open(file[i])[0].header
        f = flat.astype(np.float32)
        mean, median, std = sigma_clipped_stats(f, cenfunc='median', stdfunc='mad_std', sigma=3)
        l_final = l_hdu.astype(np.float32) / (f / median)
        fits.writeto(path + '/'+color+'_pp/pp'+obj_name+str(n)+'_'+color+'.fits', l_final.astype(np.float32), header=l_hdr, overwrite=True)
        bar1.update(i)
    bar1.finish()

def astrometry(path, obj_name, ra, dec, radius):
    file = open(path+'/'+obj_name+'.sh', 'w')
    file.write(f'solve-field --index-dir /Users/jang-in-yeong/solve/index4100 --use-source-extractor -3 {ra} -4 {dec} -5 {radius} --no-plots *.fits \nrm -rf *.xyls *.axy *.corr *.match *.new *.rdls *.solved\nulimit -n 4096')
    file.close()

from convert_fits import convert_fits
def split_rgb(path, obj_name):
    convert_fits.debayer_RGGB_multi(path)
    convert_fits.split_rgb_multi(path, obj_name)

import time
from sky_sub_color import sky_sub
def process(path, obj_name):
    start_time = time.time()
    Fits.mkdir(path, '/process')
    db_sub(path, obj_name)
    convert_fits.debayer_RGGB_multi(path)
    convert_fits.split_rgb_multi(path, obj_name)
    color_list = ['r', 'g', 'b']
    for i in color_list:
        Master.masking(path, i)
        Master.dark_sky_flat(path, i)
        flat_corr(path, obj_name, i)
        sky_sub(path, obj_name, i)
    end_time = time.time()
    print(f'{end_time - start_time} seconds') 

def binning(data, bin):
    img_height, img_width = data.shape

    newImage = np.zeros((bin,bin), dtype=np.float32)

    new_height = img_height//bin
    new_width = img_width//bin
    """
    binning
    """
    for j in range(bin):
        for i in range(bin):
            y = j*new_height
            x = i*new_width
            pixel = data[y:y+new_height, x:x+new_width]
            newImage[j,i] = np.nanmedian(pixel).astype(np.float32)
    return newImage.astype(np.float32)

#process('/volumes/ssd/NGC5907/2', 'NGC5907')
import ray
file = glob.glob('/volumes/ssd/NGC5907/1/r_pp/pp*.fits')
@ray.remote
def bin(file, i):
    n = format(i, '04')
    hdu = fits.open(file[i])[0].data 
    b_hdu = binning(hdu, 1504)
    fits.writeto('/volumes/ssd/NGC5907/1/r_pp/binned_NGC5907'+str(n)+'.fits', b_hdu, overwrite=True)

#ray.get([bin.remote(file, i) for i in range(len(file))])
