from photutils.background import Background2D, MedianBackground
import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt
from photutils.segmentation import SourceFinder
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
import warnings
from astropy.modeling import models, fitting
import astropy.io.fits as fits
from astropy.stats import sigma_clipped_stats
import sep
from scipy.ndimage import binary_dilation
from mask import region_mask

def mask(single_name):
        data1 = fits.open(single_name)[0].data
        bkg_estimator = MedianBackground()
        bkg = Background2D(data1, (64,64), filter_size=(3,3), bkg_estimator=bkg_estimator)
        data = data1 - bkg.background
        mena, median, std = sigma_clipped_stats(bkg.background, cenfunc='median', stdfunc='mad_std', sigma=3)
        threshold = median + 3*std

        kernel = make_2dgaussian_kernel(1.05, size=13)
        convolved_data = convolve(data, kernel)

        finder = SourceFinder(npixels=10, progress_bar=False)
        segment_map = finder(convolved_data, threshold)

        mask_map = np.array(segment_map)
        smoothed = convolve(mask_map, kernel)

        masked = np.where((smoothed!=0), np.nan, data1)
        return smoothed #masked

def se_mask(data):
        #data = fits.open(single_name)[0].data
        data1 = data.astype(data.dtype.newbyteorder('='))
        bkg = sep.Background(data1)
        bkg_rms = bkg.rms()
        mean, median, std = sigma_clipped_stats(bkg_rms, cenfunc='median', stdfunc='mad_std', sigma=3)
        subd = data - bkg
        obj, seg_map = sep.extract(subd, 3.0*bkg.globalrms, segmentation_map=True) 
        mask_map = np.array(seg_map)
        kernel = np.array([[1,1,1],[1,1,1],[1,1,1]]) #skimage.morphology.disk(3)
        mask_map_d = binary_dilation(mask_map, kernel, iterations=2)
        masked = np.where((mask_map_d!=0), np.nan, data)
        return masked.astype(np.float32)

def sky_model(data, bin):
        img_height, img_width = data.shape

        newImage = np.zeros((bin,bin), dtype=data.dtype)

        new_height = img_height//bin
        new_width = img_width//bin

        """
        the center position of binned pixel
        """
        xx_m = np.arange(0,img_width, img_width/bin)
        yy_m = np.arange(0, img_height, img_height/bin)

        x_m = np.array([[i for i in xx_m] for j in yy_m])
        y_m = np.array([[j for i in xx_m] for j in yy_m])

        """
        binning
        """
        for j in range(bin):
            for i in range(bin):
                y = j*new_height
                x = i*new_width
                pixel = data[y:y+new_height, x:x+new_width]
                newImage[j,i] = np.median(pixel[~np.isnan(pixel)])
                
        """
        calculate matrix x and y, these are positon component or img
        """        
        x1 = np.array([[i for i in range(img_width)] for j in range(img_height)])
        y1 = np.array([[j for i in range(img_width)] for j in range(img_height)])

        data_nc = np.ma.masked_invalid(newImage)

        """
        modeling
        """

        p_init = models.Polynomial2D(degree=2) #다항함수 모델링
        fit_p = fitting.LinearLSQFitter()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = fit_p(p_init, x_m, y_m, data_nc) #하늘의 모델을 반환(x,y)
        return model(x1, y1)

def sub(data, sky):
        sub = np.array(data) - np.array(sky)
        return sub

import progressbar

def sky_sub(path, obj_name, color):
      import glob
      import os
      if not os.path.exists(path + '/sky_subed'):
        os.mkdir(path + '/sky_subed_'+color)
      p = glob.glob(path + '/'+color+'_pp/pp*.fits')
      m = glob.glob(path + '/'+color+'/mask/*.fits')
      bar1 = progressbar.ProgressBar(maxval=len(p), widgets=['[',progressbar.Timer(),']',progressbar.Bar()]).start()
      for i in range(len(p)):
        n = format(i, '04')
        input = p[i]
        mask_i = m[i]
        hdr = fits.open(input)[0].header
        data = fits.open(input)[0].data
        mask = fits.open(mask_i)[0].data
        data1 = np.where(mask!=0,np.nan, data)
        sky = sky_model(data1, 64).astype(np.float32)
        subed = (data - sky).astype(np.float32)
        hdr.append(('sky_sub', 'Python', 'sky subtraction' ))
        fits.writeto(path +'/sky_subed_'+color+'/pp' + obj_name + str(n)+'.fits',subed , header=hdr, overwrite=True)
        bar1.update(i)
      bar1.finish()
        
      
def astrometry(path, obj_name, ra, dec, radius):
    file = open(path+'/'+obj_name+'.sh', 'w')
    file.write(f'solve-field --index-dir /Users/jang-in-yeong/solve/index4100 --use-source-extractor -3 {ra} -4 {dec} -5 {radius} --no-plots *.fits \nrm -rf *.xyls *.axy *.corr *.match *.new *.rdls *.solved\nulimit -n 4096')
    file.close()

warnings.filterwarnings('ignore')

def model_plot(path):
     hdu = fits.open(path)[0].data 
     mask = region_mask(hdu, 1.5)
     masked = np.where(mask!=0, np.nan, hdu)
     sky = sky_model(masked, 32,2)
     plt.imshow(sky, cmap='grey', origin='lower')
     plt.colorbar()
     plt.xlabel('x')
     plt.ylabel('y')
     plt.title('Bkg Model')
     plt.show()

def mask_plot(path):
     hdu = fits.open(path)[0].data
     mask = region_mask(hdu, 1.5)
     masked = np.where(mask!=0,np.nan, hdu)
     median = np.nanmedian(masked)
     std = np.nanstd(masked)
     plt.imshow(masked, vmax=median+3*std, vmin=median-3*std, origin='lower')
     plt.colorbar()
     plt.title('Masked Image')
     plt.xlabel('x')
     plt.ylabel('y')
     plt.show()

def save_mask(path):
     hdu = fits.open(path)[0].data
     mask = region_mask(hdu,1.5)
     masked = np.where(mask!=0, np.nan, hdu)
     fits.writeto('/volumes/ssd/intern/25_summer/M101_L/pp_masked_nrm.fits', masked, overwrite=True)

def save_model(path):
     masked = se_mask(path)
     sky = sky_model(masked, 64)
     fits.writeto('/volumes/ssd/intern/25_summer/M101_L/bkg.fits', sky, overwrite=True)

#sky_sub('/volumes/ssd/NGC5907/1', 'NGC5907', 'b')
#astrometry('/volumes/ssd/intern/25_summer/NGC891_r/sky_subed','NGC891','02:22:32.9','+42:20:54.0','1.5')
#save_mask('/volumes/ssd/intern/25_summer/M101_L/sky_subed/ppM1010000.fits')
#save_model('/volumes/ssd/intern/25_summer/M101_L/pp/ppM101_0000.fits')
#model_plot('/volumes/ssd/intern/25_summer/M101_L/pp/ppM101_0000.fits')
#mask_plot('/volumes/ssd/intern/25_summer/M101_L/pp/ppM101_0024.fits')