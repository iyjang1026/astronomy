import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt
import warnings
from astropy.modeling import models, fitting
import astropy.io.fits as fits
from astropy.stats import sigma_clipped_stats
import ray
import sys

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
            newImage[j,i] = np.nanmedian(pixel)
                
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

import progressbar

from mask1 import masking
def sky_sub(path, obj_name):
    import glob
    import os
    """
    if not os.path.exists(path + '/sky_subed'):
        os.mkdir(path + '/sky_subed')
    """
    p = sorted(glob.glob(path + '/*.fit'))
    #m = sorted(glob.glob(path + '/'+color+'/mask/*.fits'))
    bar1 = progressbar.ProgressBar(maxval=len(p), widgets=['[',progressbar.Timer(),']',progressbar.Bar()]).start()
    #@ray.remote
    def sky(p, i):
        n = format(i, '04')
        input = p[i]
        #mask_i = m[i]
        hdr = fits.open(input)[0].header
        data = fits.open(input)[0].data
        #mask = fits.open(mask_i)[0].data
        #data1 = np.where(mask!=0,np.nan, data)
        sky = sky_model(data, 64).astype(np.float32)
        subed = (data - sky).astype(np.float32)
        #hdr.append(('sky_sub', 'Python', 'sky subtraction' ))
        #fits.writeto(path +'/sky_subed/no_bkg' + obj_name + str(n)+'_.fits',subed , header=hdr, overwrite=True)
        bar1.update(i)
        return data, sky, subed
    
    fig, ax = plt.subplots(3,2, figsize=(10,8))
    title = ['BSHS', 'UOU']
    for i in range(len(p)):
        data, sky1, subed = sky(p,i)
        m1, med1, std1 = sigma_clipped_stats(data, cenfunc='median',stdfunc='mad_std', sigma=3.)
        m2, med2, std2 = sigma_clipped_stats(subed, cenfunc='median',stdfunc='mad_std', sigma=3.)
        ax[0,i].imshow(data.astype(np.float32), origin='lower', vmax=med1+3*std1, vmin=med1-3*std1)
        ax[0,i].set_title(title[i]+' raw')
        ax[0,i].set_xlabel('x')
        ax[0,i].set_ylabel('y')
        ax[1,i].imshow(sky1.astype(np.float32), origin='lower')
        ax[1,i].set_title('Bkg Model')
        ax[1,i].set_xlabel('x')
        ax[1,i].set_ylabel('y')
        ax[2,i].imshow(subed.astype(np.float32), origin='lower', vmax=med2+3*std2, vmin=med2-3*std2)
        ax[2,i].set_title('Bkg Subed')
        ax[2,i].set_xlabel('x')
        ax[2,i].set_ylabel('y')
    plt.show()
    
    """
    ray.get([sky.remote(p,i) for i in range(1)])
    ray.shutdown()
    bar1.finish()
    """    
warnings.filterwarnings('ignore')

sky_sub('/volumes/ssd/satelite/plot','plot')

