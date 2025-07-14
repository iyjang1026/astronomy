import astropy.io.fits as fits
from astropy.convolution import convolve
import cv2
import numpy as np
import glob
import os
import time
def fits2jpg(path, obj_name):
    start_time = time.time()
    if not os.path.exists(path + '/jpg'):
        os.mkdir(path + '/jpg')
    kernel = np.array([[0,1,0],[1,1,1],[0,1,0]])
    file = glob.glob(path+'/*.fits')
    for i in range(len(file)):
        n = format(i, '04')
        hdu = fits.open(file[i])[0].data 
        conv_hdu = convolve(hdu, kernel)
        cv2.imwrite(path+'/jpg/conv_80_'+obj_name+'_'+str(n)+'.jpg', conv_hdu.astype(np.float32), [cv2.IMWRITE_JPEG_QUALITY, 80])
    end_time = time.time()
    eta = end_time-start_time
    print(f'{eta//60} min {eta-(eta//60)*60} seconds')

fits2jpg('/volumes/ssd/intern/25_summer/M101_L/sky_subed', 'M101')