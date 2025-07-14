import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.convolution import convolve
from photutils.segmentation import detect_sources, make_2dgaussian_kernel, SourceCatalog, deblend_sources
from photutils.background import MedianBackground, Background2D
from photutils.aperture import EllipticalAperture
import ray

def region_mask(hdu):
    #hdu = fits.open(path)[0].data 
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
    arr_zero = np.zeros_like(hdu).astype(np.float32)
    mask_x, mask_y = mask.shape
    
    st_x = np.int16(xy[1] - mask_x/2)
    st_y = np.int16(xy[0] - mask_y/2)
    
    arr_zero[st_x:st_x+mask_x,st_y:st_y+mask_y] = mask[:mask_x,:mask_y]
    seg = np.array(seg_map)
    masked_map = np.where(seg!=0, 1, 0) + arr_zero
    masked = np.where(masked_map!=0, np.nan, hdu).astype(np.float32)
    return masked



hdu = fits.open('/volumes/ssd/intern/25_summer/M101_L/pp_obj/ppM101_0000.fit')[0].data
arr_ref = ray.put(hdu)



#plt.imshow(mask, origin='lower')
#plt.show()