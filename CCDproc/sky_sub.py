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
                newImage[j,i] = np.nanmedian(pixel) #np.ma.median(pixel) #
                
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

def sky_sub(hdu, hdr, mask):
        #mask =  fits.open(mask)[0].data #region_mask(hdu,1.5, 0.8) #
        m_data = np.where(mask!=0,np.nan, hdu) #np.ma.masked_where(mask, hdu) #
        sky = sky_model(m_data, 64).astype(np.float32)
        subed = np.array(hdu-sky).astype(np.float32)
        hdr.append(('sky_sub', 'Python', 'sky subtraction' ))
        return np.array(subed).astype(np.float32), hdr
        
      
def astrometry(path, obj_name, ra, dec, radius):
    file = open(path+'/'+obj_name+'.sh', 'w')
    file.write(f'solve-field --index-dir /Users/jang-in-yeong/solve/index4100 --use-source-extractor -3 {ra} -4 {dec} -5 {radius} --no-plots *.fits \nrm -rf *.xyls *.axy *.corr *.match *.new *.rdls *.solved\nulimit -n 4096')
    file.close()

warnings.filterwarnings('ignore')
fig, ax = plt.subplots(1,3, figsize=(12,4))
def model_plot(path):
     hdu = fits.open(path)[0].data 
     mask = fits.open('/volumes/ssd/intern/25_summer/M101_L/mask/mask0000.fits')[0].data ##
     masked = np.where(mask!=0, np.nan, hdu) #
     sky = sky_model(masked, 64)
     mask1 = region_mask(hdu, 1.5, 0.8)
     masked1 = np.where(mask1!=0, np.nan, hdu)
     sky1 = sky_model(masked1, 64)
     model_img = ax[2].imshow(sky1, origin='lower')
     plt.colorbar(mappable=model_img, ax=ax[2])
     ax[2].set_xlabel('x')
     ax[2].set_ylabel('y')
     ax[2].set_title('Bkg Model')
     

def mask_plot(path):
     hdu = fits.open(path)[0].data
     mask = fits.open('/volumes/ssd/intern/25_summer/M101_L/mask/mask0000.fits')[0].data #region_mask(hdu, 1.5, 0.8)
     masked = np.where(mask!=0,np.nan, hdu) #np.ma.masked_where(mask, hdu) #
     median = np.nanmedian(masked)
     std = np.nanstd(masked)
     img = ax[0].imshow(hdu,vmax=median+3*std, vmin=median-3*std, origin='lower') # 
     mask = ax[1].imshow(mask, origin='lower')
     ax[1].set_title('Mask')
     ax[1].set_xlabel('x')
     ax[1].set_ylabel('y')
     #plt.colorbar(mappable=img, ax=ax[0])
     ax[0].set_title('Preprocessed Image')
     ax[0].set_xlabel('x')
     ax[0].set_ylabel('y')
"""
model_plot('/volumes/ssd/intern/25_summer/M101_L/pp/ppM101_0000.fits')
mask_plot('/volumes/ssd/intern/25_summer/M101_L/pp/ppM101_0000.fits')
plt.show()
"""
def save_mask(path):
     hdu = fits.open(path)[0].data
     mask = region_mask(hdu,1.5, 0.9)
     #masked = np.where(mask!=0, np.nan, hdu)
     fits.writeto('/volumes/ssd/intern/25_summer/M101_L/mask1.fits', mask, overwrite=True)

def save_model(path):
     hdu = fits.open(path)[0].data
     m_hdu = fits.open('/volumes/ssd/intern/25_summer/M101_L/pp_obj/ppM101_0024.fit')[0].data
     mask = region_mask(m_hdu, 1.5, 0.8)#fits.open('/volumes/ssd/intern/25_summer/M101_L/mask/mask0024.fits')[0].data #
     masked = np.where(mask!=0, np.nan, hdu)
     sky = sky_model(masked, 64)
     fits.writeto('/volumes/ssd/intern/25_summer/M101_L/bkg_3.fits', sky, overwrite=True)

#sky_sub('/volumes/ssd/intern/25_summer/M101_L', 'M101')
#astrometry('/volumes/ssd/intern/25_summer/NGC891_r/sky_subed','NGC891','02:22:32.9','+42:20:54.0','1.5')
#save_mask('/volumes/ssd/intern/25_summer/M101_L/pp/ppM101_0024.fits')
#save_model('/volumes/ssd/intern/25_summer/M101_L/pp/ppM101_0024.fits')
#model_plot('/volumes/ssd/intern/25_summer/M101_L/pp/ppM101_0024.fits')
#mask_plot('/volumes/ssd/intern/25_summer/M101_L/sky_subed/ppM1010000.fits')