import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.convolution import convolve
from astropy.stats import sigma_clipped_stats
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

def region_mask(hdu, thrsh, eps_thr):
    half = disk(100)
    z_arr = np.zeros_like(hdu)
    z_arr[2048-100:2048,1212-100-1:1212+100] += half[0:100,:]
    bkg_est = MedianBackground()
    bkg = Background2D(hdu, (64,64), filter_size=(5,5), bkg_estimator=bkg_est, mask=z_arr)
    data = hdu - bkg.background
    threshold = thrsh*bkg.background_rms
    kernel = make_2dgaussian_kernel(fwhm=3.0, size=5)
    conv_hdu = convolve(data, kernel)
    seg_map = detect_sources(conv_hdu, threshold, npixels=5, mask=z_arr)
    segm_deblend = deblend_sources(conv_hdu, seg_map,
                               npixels=2000,connectivity=8, mode='exponential', nlevels=32, contrast=0.001,
                               progress_bar=False)
    seg = np.array(seg_map)
    
    cat = SourceCatalog(data, segm_deblend, convolved_data=conv_hdu)
    """
    ap = cat.kron_aperture
    l = [x for x in ap if x!=None]
    a_list = []
    for i in l:
        a = None
        b = None
        eps = None
        a = i.a
        b = i.b
        eps = np.sqrt(1-(b/a)**2)
        if eps > eps_thr:
            a_list.append(0)
        else:
            a_list.append(b)
    """
    a_list = list(cat.semimajor_sigma.value)
    arr_zero = np.zeros_like(hdu).astype(np.float32) 
    tmp = a_list.copy()
    tmp.sort()
    tmp_num = tmp[-20:]
    top_idx = [a_list.index(x) for x in tmp_num]
    for i in top_idx:
        """
        g_aper = l[i]
        a = g_aper.a
        b = g_aper.b
        xypos = g_aper.positions
        theta = g_aper.theta
        xy = (int(xypos[0]), int(xypos[1]))
        """
        cat0 = cat[i]
        xy = (cat0.xcentroid, cat0.ycentroid)
        theta = cat0.orientation.value *np.pi /180
        a,b = 3*cat0.semimajor_sigma.value, 3*cat0.semiminor_sigma.value
        aperture = EllipticalAperture(xy, 3*a, 3*b, theta)
        #aperture = #EllipticalAperture(xy, 3.5*a, 3.5*b, theta=theta)
        mask = np.array(aperture.to_mask(method='center')).astype(np.int8)
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
    
    kernel0 = disk(3) 
    seg_d= binary_dilation(seg, kernel0, iterations=1)
    masked_map = np.where(seg_d!=0, 1, 0) + arr_zero
    half = disk(100)
    masked_map[2048-100:2048,1212-100-1:1212+100] += half[0:100,:]
    masked = np.where(masked_map!=0, 1, 0).astype(np.int8)
    
    return np.array(masked, dtype=np.int8)

"""
hdu = fits.open('/volumes/ssd/intern/25_summer/M51_L/pp_obj/ppM51_0000.fit')[0].data
x,y = hdu.shape
mask = region_mask(hdu,1.5, 0.8)
map = np.where(mask!=0, np.nan, hdu)
#fits.writeto('/volumes/ssd/intern/25_summer/M101_L/pp_mask_nrm_test_coadd.fits', map, overwrite=True)
#map1 = np.where(map==0,np.nan, map)
median = np.nanmedian(map)
std = np.nanstd(map)
plt.imshow(map,vmax=median+3*std, vmin=median-3*std, origin='lower')
#plt.imshow(hdu, origin='lower') #vmax=median+3*std, vmin=median-3*std,
#plt.imshow(map,vmax=median+3*std, vmin=median-3*std ,origin='lower')
#plt.imshow(map[int(x/2-1300):int(x/2+1300),int(y/2-1300):int(y/2+1300)],vmax=median+3*std, vmin=median-3*std,
 #           origin='lower')
plt.colorbar()
plt.show()
"""