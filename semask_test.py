import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.convolution import convolve
from photutils.segmentation import SourceCatalog, deblend_sources,detect_sources, SegmentationImage, make_2dgaussian_kernel
from photutils.background import MedianBackground, Background2D
from photutils.aperture import EllipticalAperture
from multiprocessing import freeze_support
import sys
import sep
from skimage.morphology import disk

def mask(hdu, thr):
    #hdu = hdu.astype(hdu.dtype.newbyteorder('='))
    """
    bkg = sep.Background(hdu)
    bkg_rms = bkg.globalrms
    subed_data = hdu - bkg
    objs, seg = sep.extract(subed_data, thr, err=bkg_rms, segmentation_map=True)
    segm = SegmentationImage(seg)
    """
    half = disk(100)
    z_arr = np.zeros_like(hdu)
    z_arr[2048-100:2048,1212-100-1:1212+100] += half[0:100,:]
    bkg_est = MedianBackground()
    bkg = Background2D(hdu, (64,64), filter_size=(5,5), bkg_estimator=bkg_est, mask=z_arr)
    data = hdu - bkg.background
    threshold = thr*bkg.background_rms
    kernel = make_2dgaussian_kernel(fwhm=3.0, size=5)
    conv_hdu = convolve(data, kernel)
    seg_map = detect_sources(conv_hdu, threshold, npixels=5, mask=z_arr)
    deblended_seg = deblend_sources(conv_hdu, seg_map, npixels=1500, mode='exponential', nproc=4, connectivity=8)
    mean, median, std = sigma_clipped_stats(hdu, cenfunc='median', stdfunc='mad_std', sigma=3)
    plt.imshow(hdu,vmax=median+3*std, vmin=median-3*std, origin='lower')
    cat_o = SourceCatalog(hdu, convolved_data=conv_hdu, segment_img=deblended_seg)
    a_list = list(cat_o.semimajor_sigma.value)
    tmp = a_list.copy()
    tmp.sort()
    tmp_num = tmp[-20:]
    top_idx = [a_list.index(x) for x in tmp_num]
    
    for i in top_idx:
        cat = cat_o[i]
        theta = cat.orientation.value *np.pi /180
        a,b = 3*cat.semimajor_sigma.value, 3*cat.semiminor_sigma.value
        aper = EllipticalAperture((cat.xcentroid, cat.ycentroid), 2.5*a, 2.5*b, theta)
        aper.plot(color='C3')
    plt.show()

hdu = fits.open('/volumes/ssd/intern/25_summer/NGC6946_L/pp_obj/ppNGC6946_0000.fit')[0].data 

if __name__=='__main__':
    freeze_support()
    mask(hdu, 1.5)