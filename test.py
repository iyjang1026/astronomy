import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats, sigma_clip
from astropy.wcs import WCS
from astropy.modeling import models, fitting
from astropy.stats import sigma_clipped_stats
import glob
import sys
from mask1 import region_mask
def coadd_plot(path):
    hdul = fits.open(path)
    hdu = hdul[0].data 
    hdr = hdul[0].header
    wcs = WCS(hdr)
    mean, median, std = sigma_clipped_stats(hdu, cenfunc='median', stdfunc='mad_std', sigma=3)
    fig, ax = plt.subplots(subplot_kw=dict(projection=wcs))
    ax.imshow(hdu, vmax=median+3*std, vmin=median-3*std, cmap='grey', origin='lower')
    ax.set(xlabel='R.A.', ylabel='Dec')
    ax.set_title('sep masking flat, model processed')
    print(std)
    plt.show()


def hist(path):
    file = glob.glob(path + '/pp_masked_nrm*.fits')
    print(file)
    bin = 1024
    sampling_size = 1000
    name = ['rm_rm_single', 'sep_sep_single', 'rm_rm_coadd','sep_sep_coadd','sep_rm_single', 'sep_rm_coadd']
    for i in range(len(file)):
        hdu = fits.open(file[i])[0].data
        x,y = hdu.shape
        std_data = hdu[int(x/2-sampling_size/2):int(x/2+sampling_size/2),
                       int(y/2-sampling_size/2):int(y/2+sampling_size/2)]
        hdu1 = hdu[~np.isnan(hdu)].flatten()
        hdu1.astype(np.float32)
        hdu2, bin_edges2 = np.histogram(hdu1, bins=bin, range=(-10000,10000))
        x1 = np.resize(bin_edges2, (bin,))
        norm_count = (hdu2 - np.max(hdu2)) / np.max(hdu2)
        m, medi, std1 = sigma_clipped_stats(std_data, cenfunc='median', stdfunc='mad_std', sigma=3)
        label = name[i]
        print(f'std of {label} is {std1}')
        plt.plot(x1, norm_count, label=label)

    #plt.hist(hdu2, bins=x1, label='Histogram')
    #plt.plot(x1, g(x1), label='fitted')
    
    plt.legend()
    plt.title('Histogram of each process')
    plt.xlabel('Level(ADU)')
    plt.ylabel('Normalized Count')
    plt.xlim(-1500,1500)
    #plt.show()

def image_plot(path):
    preprocessed = glob.glob(path + '/pp/pp*.fits')[0]
    mask = path + '/pp_masked.fits'
    model = path + '/bkg.fits'
    sky_subed = path + '/masked.fits'
    f_list = np.array([[preprocessed, mask],[model, sky_subed]])
    title = np.array([['Preprocessed','Masked'],['Bkg model', 'Sky subed']])
    x, y = f_list.shape
    fig, ax = plt.subplots(2,2)
    for i in range(2):
        for j in [0]:
            hdul = fits.open(f_list[i,j])
            hdu = hdul[0].data
            hdu = hdu[:,1024]
            mean, median, std = sigma_clipped_stats(hdu, cenfunc='median', stdfunc='mad_std',sigma=3)
            #norm_hdu = (hdu - median)/median
            #mean, median1, std1 = sigma_clipped_stats(norm_hdu, cenfunc='median', stdfunc='mad_std',sigma=3)
            #ax[i,j].imshow(norm_hdu, vmax=median1+3*std1, vmin=median1-3*std1,origin='lower') #
            x1 = np.linspace(0,2048,2048)
            ax[i,j].plot(x1, hdu)
            ax[i,j].set_title(str(title[i,j]))
            ax[i,j].set_xlabel('x')
            ax[i,j].set_ylabel('Level(ADU)')
    for i in range(2):
        for j in [1]:
            hdul = fits.open(f_list[i,j])
            hdu = hdul[0].data
            hdu = hdu[:,1024]
            mean, median, std = sigma_clipped_stats(hdu, cenfunc='median', stdfunc='mad_std',sigma=3)
            hdu1 = sigma_clip(hdu, cenfunc='median', stdfunc='mad_std', sigma=3)
            #norm_hdu = (hdu - median)/median
            #mean, median1, std1 = sigma_clipped_stats(norm_hdu, cenfunc='median', stdfunc='mad_std',sigma=3)
            #ax[i,j].imshow(norm_hdu, vmax=median1+3*std1, vmin=median1-3*std1,origin='lower') #
            x1 = np.linspace(0,2048,2048)
            g_init = models.Polynomial1D(degree=1)
            fit_g = fitting.TRFLSQFitter()
            hdu_flatten = hdu1[~np.isnan(hdu1)].flatten()
            x2 = np.linspace(0,hdu_flatten.shape[0], hdu_flatten.shape[0])
            g = fit_g(g_init, x2, hdu_flatten)
            #ax[i,j].scatter(x1,hdu1,c='grey', s=3)
            #ax[i,j].scatter(x1,hdu, c='tomato', s=3)
            ax[i,j].plot(x2, g(x2), c='black')
            ax[i,j].set_title(str(title[i,j]))
            tan = (g(x2)[-1]-g(x2)[0])/ x2[-1]-x2[0]
            ax[i,j].text(500,median-3*std, f'tan={tan:.4f}\nstd={std:.4f}')
            ax[i,j].set_ylim(-40,40) #(median-3.5*std, median+3.5*std)
            ax[i,j].set_ylabel('Level(ADU)')
            ax[i,j].set_xlabel('x')
    plt.show()

def residual(path):
    iraf = fits.open(path + '/fd.fits')[0].data 
    python_flat = fits.open(path + '/master_flat.fits')[0].data
    mean, median0, std0 = sigma_clipped_stats(iraf, cenfunc='median', stdfunc='mad_std', sigma=3)
    mean, median, std = sigma_clipped_stats(python_flat, cenfunc='median', stdfunc='mad_std', sigma=3)
    norm_iraf = iraf / median0 #(iraf - median0)/median0
    norm_python = python_flat / median #(python_flat - median)/median
    residual = norm_iraf - norm_python
    plt.imshow(residual, vmax=np.median(residual)+3*np.std(residual),vmin=np.median(residual)-3*np.std(residual)
               ,origin='lower')
    plt.colorbar()
    plt.title('region masked flat - sep masked flat')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def coadd_mask(path):
    hdu0 = fits.open(path)[0].data 
    min = np.min(hdu0)
    max = np.max(hdu0)
    norm_hdu = (hdu0 - min)/(max-min)
    mask_samp = region_mask(norm_hdu, 1.5)
    masked = np.where(mask_samp!=0, np.nan, hdu0)
    median = np.nanmedian(masked)
    std = np.nanstd(masked)
    fits.writeto('/volumes/ssd/intern/25_summer/M101_L/tset_nan0_coadd.fits', masked, overwrite=True)
    plt.imshow(masked, vmax=median+3*std, vmin=median-3*std, origin='lower')
    plt.show()
    

def img_show(path):
    fig, axes = plt.subplots(1,2)
    single = fits.open(path + '/pp_masked.fits')[0].data 
    s_median = np.nanmedian(single)
    s_std = np.nanstd(single)
    coadd = fits.open(path + '/pp_masked_coadd.fits')[0].data
    c_median = np.nanmedian(coadd)
    c_std = np.nanstd(coadd)
    axes[0].imshow(single, vmax=s_median+3*s_std, vmin=s_median-3*s_std
                  ,origin='lower')
    axes[1].imshow(coadd, vmax=c_median+3*c_std, vmin=c_median-3*c_std, origin='lower')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title('Single Masked')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_title('Coadd Masked')
    plt.show()
def fig():
    path = '/volumes/ssd/intern/25_summer/M101_L'
    fig, axes = plt.subplots(1,2)
    file = glob.glob(path + '/pp_masked_nrm_coadd*.fits')
    for i in range(len(file)):
        hdu = fits.open(file[i])[0].data 
        mean, median, std = sigma_clipped_stats(hdu, cenfunc='median', stdfunc='mad_std', sigma=3)
        axes[i].imshow(hdu, vmax=median+3*std, vmin=median-3*std, origin='lower')
        if file[i] == path +'/pp_masked_nrm_coadd.fits':
            title = 'Previous masking Coadd Image'
        else:
            title = 'New masking Coadd Image'
        axes[i].set_title(title)
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
    plt.show()
def std(arr):
    import ray
    import time
    std_list = []
    @ray.remote
    def random_samp(arr, samp_size):
        x, y = arr.shape
        x_num = np.random.randint(0,x-samp_size)
        y_num = np.random.randint(0,y-samp_size)
        samp_arr = arr[x_num:x_num+samp_size,y_num:y_num+samp_size]
        mean, median, std = sigma_clipped_stats(samp_arr, cenfunc='median',stdfunc='mad_std',sigma=3)
        return std
    #path = '/volumes/ssd/intern/25_summer/M101_L'
    #hdu0 = fits.open(path+'/pp_masked_nrm.fits')[0].data
    x,y = arr.shape
    cx, cy = int(x/2), int(y/2)
    hdu = arr[cx-1000:cx+1000, cy-1000:cy+1000]
    st_time = time.time()
    std_list.append(ray.get([random_samp.remote(hdu,200) for i in range(2000)]))
    std_arr = np.array(std_list)
    mean, median, std = sigma_clipped_stats(std_arr, cenfunc='median', stdfunc='mad_std', sigma=3)
    end_time = time.time()
    ray.shutdown()
    eta = end_time - st_time
    print(median,f'{eta//60}mins {eta-(eta//60)}seconds')
#coadd_plot('/volumes/ssd/intern/25_summer/M101_L/sky_sub/coadd.fits')
#hist('/volumes/ssd/intern/25_summer/M101_L')
#image_plot('/volumes/ssd/intern/25_summer/M101_L/')
#residual('/volumes/ssd/intern/25_summer/M101_L/process')
#coadd_mask('/volumes/ssd/intern/25_summer/test_nan0.fits')
#img_show('/volumes/ssd/intern/25_summer/M101_L')
#fig()
import warnings
warnings.filterwarnings('ignore')

#hdu = fits.open('/volumes/ssd/intern/25_summer/M101_L/pp_masked_nrm.fits')[0].data 
#std(hdu)
"""
hdu = fits.open('/volumes/ssd/intern/25_summer/M101_L/sky_subed/coadd.fits')[0].data 
mean, median, std = sigma_clipped_stats(hdu, cenfunc='median', stdfunc='mad_std', sigma=3)
plt.imshow(hdu, vmax=median+3*std, vmin=median-3*std, origin='lower', cmap='grey')
plt.show()
"""