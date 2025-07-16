import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.convolution import convolve
from photutils.segmentation import detect_sources, make_2dgaussian_kernel, SourceCatalog, deblend_sources
from photutils.background import MedianBackground, Background2D
from photutils.aperture import EllipticalAperture
from scipy.ndimage import binary_dilation
from skimage.morphology import disk
import sys
import sep
def se_mask(arr):
        data = np.array(arr)
        data1 = data.astype(data.dtype.newbyteorder('='))
        bkg = sep.Background(data1)
        subd = data - bkg
        obj, seg_map = sep.extract(subd, 1.5*bkg.globalrms, segmentation_map=True)
        mask_map = np.array(seg_map)
        kernel = np.array([[1,1,1],[1,1,1],[1,1,1]]) #skimage.morphology.disk(3)
        mask_map_d = binary_dilation(mask_map, kernel, iterations=2)
        masked = np.where((mask_map_d!=0), np.nan, data)
        return masked.astype(np.float32)
    
def masking(arr):
    bkg_estimator = MedianBackground()
    bkg = Background2D(arr, (64,64), filter_size=(3,3), bkg_estimator=bkg_estimator)
    data = arr - bkg.background
    threshold = 3 * bkg.background_rms
    kernel = make_2dgaussian_kernel(3.0, size=5)
    convolved_data = convolve(data, kernel)
    seg_map = detect_sources(convolved_data, threshold, npixels=10)
    mask_map = np.array(seg_map)
    kernel = np.array([[1,1,1],[1,1,1],[1,1,1]]) #skimage.morphology.disk(3)
    mask_map_d = binary_dilation(mask_map, kernel, iterations=2)
    masked = np.where((mask_map_d!=0), np.nan, arr)
    return masked.astype(np.float32)

def region_mask(hdu, thrsh):
    bkg_est = MedianBackground()
    bkg = Background2D(hdu, (64,64), filter_size=(3,3), bkg_estimator=bkg_est)
    data = hdu - bkg.background
    threshold = thrsh*bkg.background_rms
    kernel = make_2dgaussian_kernel(fwhm=3.0, size=5)
    conv_hdu = convolve(data, kernel)
    seg_map = detect_sources(conv_hdu, threshold, npixels=7)
    segm_deblend = deblend_sources(conv_hdu, seg_map,
                               npixels=80, nlevels=32, contrast=0.001,
                               progress_bar=False)
    seg = np.array(seg_map)
    
    cat = SourceCatalog(data, segm_deblend, convolved_data=conv_hdu)
    
    
    ap = cat.kron_aperture
    a_list = []
    for i in ap:
        a = None
        b = None
        eps = None
        a = i.a
        b = i.b
        eps = np.sqrt(1-(b/a)**2)
        if eps > 0.8:
            a_list.append(0)
        else:
            a_list.append(b)
   
    
    arr_zero = np.zeros_like(hdu).astype(np.float32) 
    tmp = a_list.copy()
    tmp.sort()
    tmp_num = tmp[-20:]
    top_idx = [a_list.index(x) for x in tmp_num]
    #plt.imshow(hdu, origin='lower')
    for i in top_idx:
        g_aper = ap[i]
        a = g_aper.a
        b = g_aper.b
        xypos = g_aper.positions
        theta = g_aper.theta
        xy = (int(xypos[0]), int(xypos[1]))
        aperture = EllipticalAperture(xy, 3.5*a, 3.5*b, theta=theta)
        mask = np.array(aperture.to_mask(method='center')).astype(np.int8)
        #aperture.plot(color='C3')
        mask_x, mask_y = mask.shape
    
        st_x = np.int16(xy[1] - mask_x/2)
        st_y = np.int16(xy[0] - mask_y/2)
    
        x, y = hdu.shape
   
        def lim(st, mask_s, arr_s):
            if st < 0 and st+mask_s<arr_s:
                arr_st = 0
                mask_st = -st
                mask_l = mask_s
            elif st<0 and st+mask_s>arr_s:
                arr_st = 0
                mask_st = -st
                mask_l = mask_s + st - arr_s
        
            elif st+mask_s > arr_s:
                arr_st = st
                mask_st = 0
                mask_l = arr_s - st

            else:
                arr_st = st
                mask_st = 0
                mask_l = mask_s
            return arr_st, mask_st, mask_l
        
        arr_x, mask_s_x, mask_l_x = lim(st_x, mask_x, x)
        arr_y, mask_s_y, mask_l_y = lim(st_y, mask_y, y)
        mask = mask[mask_s_x:mask_l_x,mask_s_y:mask_l_y] 
        m_x, m_y = mask.shape #crop mask
        arr_zero[arr_x:arr_x+m_x, arr_y:arr_y+m_y] += mask
    
    masked_map = np.where(seg!=0, 1, 0) + arr_zero #+ small_seg_d
    seg_d = np.where(masked_map!=0, 1, 0).astype(np.int8)
    kernel0 = disk(3) 
    half = disk(100)
    #seg_d[2048-100:2048,1212-100-1:1212+100] = half[0:100,:]
    masked = binary_dilation(seg_d, kernel0, iterations=3)
    return np.array(masked, dtype=np.int8)

"""
hdu = fits.open('/volumes/ssd/intern/25_summer/M101_L/pp/ppM101_0000.fits')[0].data
mask = region_mask(hdu,1.5)
map = np.where(mask!=0, np.nan, hdu)
#fits.writeto('/volumes/ssd/intern/25_summer/M101_L/pp_mask_test.fits', map, overwrite=True)
median = np.nanmedian(map)
std = np.nanstd(map)
plt.imshow(hdu, origin='lower') #vmax=median+3*std, vmin=median-3*std,
plt.imshow(map,vmax=median+3*std, vmin=median-3*std ,origin='lower')
plt.colorbar()
plt.show()
"""