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



def mask(single_name):
        data1 = fits.open(single_name)[0].data
        bkg_estimator = MedianBackground()
        bkg = Background2D(data1, (64,64), filter_size=(3,3), bkg_estimator=bkg_estimator)
        data = data1 - bkg.background

        threshold = 3 * bkg.background_rms

        kernel = make_2dgaussian_kernel(1.05, size=13)
        convolved_data = convolve(data, kernel)

        finder = SourceFinder(npixels=10, progress_bar=False)
        segment_map = finder(convolved_data, threshold)

        mask_map = np.array(segment_map)
        smoothed = convolve(mask_map, kernel)

        masked = np.where((smoothed!=0), np.nan, data1)
        return masked

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


def sky_sub(path, obj_name):
      import glob
      import os
      os.mkdir(path + '/sky_subed')
      p = glob.glob(path + '/pp*.fits')
      for i in range(len(p)):
        n = format(i, '04')
        input = p[i]
        hdr = fits.open(input)[0].header
        data = fits.open(input)[0].data
        data1 = mask(input)
        sky = sky_model(data1, 64)
        subed = sub(data,sky)
        hdr.append(('sky_sub', 'Python', 'sky subtraction' ))
        fits.writeto(path +'/sky_subed/' + obj_name + str(n)+'.fits',subed , header=hdr, overwrite=True)



#sky_sub('/volumes/ssd/2025-04-25/b', 'm106')

