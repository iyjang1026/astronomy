import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats, sigma_clip
from astropy.wcs import WCS
from astropy.modeling import models, fitting
from astropy.stats import sigma_clipped_stats
import glob
import sys
def coadd_plot(path):
    hdul = fits.open(path)
    hdu = hdul[0].data 
    hdr = hdul[0].header
    wcs = WCS(hdr)
    mean, median, std = sigma_clipped_stats(hdu, cenfunc='median', stdfunc='mad_std', sigma=3)
    fig, ax = plt.subplots(subplot_kw=dict(projection=wcs))
    ax.imshow(hdu, vmax=median+3*std, vmin=median-3*std, cmap='grey', origin='lower')
    ax.set(xlabel='R.A.', ylabel='Dec')
    ax.set_title('Coadd Image')
    print(std)
    plt.show()


def hist(path):
    file = glob.glob(path + '/pp_masked*.fits')
    bin = 1024
    for i in file:
        hdul = fits.open(i)
        hdu = hdul[0].data 
        hdul.close()
        hdu1 = hdu[~np.isnan(hdu)].flatten()
        hdu0 = np.resize(hdu1, (bin,))
        hdu1.astype(np.float32)
        print(hdu1.shape)
        #sys.exit()
        hdu2, bin_edges2 = np.histogram(hdu1, bins=bin, range=(-10000,10000))
        """
        max_idx = np.where(hdu2==np.max(hdu2))
        hist_n = np.delete(hdu2, max_idx)
        """
        x1 = np.resize(bin_edges2, (bin,))
        norm_count = (hdu2 - np.max(hdu2)) / np.max(hdu2)
        if i == path+'/pp_masked.fits':
            label = 'single_image'
        else:
            label = 'coadded_image'
        mean1, median1, std1, = sigma_clipped_stats(hdu1, cenfunc='median', stdfunc='mad_std', sigma=3)
        print(f'std of {label} is {std1}')
        plt.plot(x1, norm_count, label=label)

    #plt.hist(hdu2, bins=x1, label='Histogram')
    #plt.plot(x1, g(x1), label='fitted')
    
    plt.legend()
    plt.title('Image Histogram')
    plt.xlabel('Level(ADU)')
    plt.ylabel('Normalized Count')
    plt.xlim(-3000,3000)
    plt.show()

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
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def coadd_mask(path):
    import numpy as np
    from matplotlib import pyplot as plt
    from astropy.io import fits
    from astropy.convolution import convolve
    from photutils.segmentation import detect_sources, make_2dgaussian_kernel, SourceCatalog, deblend_sources
    from photutils.background import MedianBackground, Background2D
    from photutils.aperture import EllipticalAperture
    hdu0 = fits.open(path)[0].data 
    hdu = hdu0[600:2600,500:2500]
    bkg_est = MedianBackground()
    bkg = Background2D(hdu, (64,64), filter_size=(3,3), bkg_estimator=bkg_est)
    data = hdu - bkg.background
    threshold = 3.0*bkg.background_rms
    kernel = make_2dgaussian_kernel(fwhm=3.0, size=5)
    conv_hdu = convolve(data, kernel)
    seg_map = detect_sources(conv_hdu, threshold, npixels=10)
    segm_deblend = deblend_sources(conv_hdu, seg_map,
                               npixels=10, nlevels=32, contrast=0.001,
                               progress_bar=False)
    cat = SourceCatalog(data, segm_deblend, convolved_data=conv_hdu)
    cat.to_table()
    ap = cat.kron_aperture
    a_list = []
    for i in ap:
        a = None
        a = i.a
        a_list.append(a)
    max_idx = np.argmax(np.array(a_list).astype(np.float32))
    g_aper = ap[max_idx]
    a = g_aper.a
    b = g_aper.b
    xypos = g_aper.positions
    theta = g_aper.theta
    xy = (int(xypos[0]), int(xypos[1]))
    aperture = EllipticalAperture(xy, 5*a, 5*b, theta=theta)
    mask = np.array(aperture.to_mask(method='center')).astype(np.int8)
    arr_zero = np.zeros_like(hdu0).astype(np.float32)
    mask_x, mask_y = mask.shape
    st_x = np.int16(xy[1] - mask_x/2)
    st_y = np.int16(xy[0] - mask_y/2)
    arr_zero[st_x:st_x+mask_x, st_y:st_y+mask_y] += mask[:mask_x,:mask_y]
    seg = np.array(seg_map)
    masked = np.where(seg_map!=0, 1, 0) + arr_zero
    
    bkg_est = MedianBackground()
    bkg = Background2D(hdu0, (64,64), filter_size=(3,3), bkg_estimator=bkg_est)
    data = hdu0 - bkg.background
    threshold = 3.0*bkg.background_rms
    kernel = make_2dgaussian_kernel(fwhm=3.0, size=5)
    conv_hdu = convolve(data, kernel)
    seg_map = detect_sources(conv_hdu, threshold, npixels=10)
    arr = np.array(seg_map)
    from scipy.ndimage import binary_dilation
    kernel0 = np.array([[0,1,0],[1,1,1],[0,1,0]])
    seg_d = binary_dilation(arr, kernel0, iterations=5)
    arr0 = np.zeros_like(hdu0)
    arr0[600:2600, 500:2500] += masked[:2000,:2000]
    masked_t = np.where(seg_d!=0, 1, 0) + np.where(arr0!=2,0, 1)
    tot_masked = np.where(masked_t!=0, np.nan, hdu0)
    fits.writeto('/volumes/ssd/intern/25_summer/M101_L/pp_masked_coadd.fits', tot_masked, overwrite=True)
    plt.imshow(tot_masked, origin='lower')
    plt.show()
#coadd_plot('/volumes/ssd/intern/25_summer/M101_L/sky_subed/coadd.fits')
#hist('/volumes/ssd/intern/25_summer/M101_L')
image_plot('/volumes/ssd/intern/25_summer/M101_L/')
#residual('/volumes/ssd/intern/25_summer/M101_L/process')
#coadd_mask('/volumes/ssd/intern/25_summer/M101_L/sky_subed/coadd.fits')