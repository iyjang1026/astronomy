import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.convolution import convolve
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import detect_sources, make_2dgaussian_kernel, SourceCatalog, deblend_sources, SegmentationImage
from photutils.background import MedianBackground, Background2D
from photutils.aperture import EllipticalAperture
from scipy.ndimage import binary_dilation
from skimage.morphology import disk
import sys

def region_mask(hdu, thrsh):
    mask1 = np.where(hdu!=0, False, True)
    bkg_est = MedianBackground()
    bkg = Background2D(hdu, (64,64), filter_size=(3,3), bkg_estimator=bkg_est, mask=mask1)
    data = hdu - bkg.background
    threshold = thrsh*bkg.background_rms
    kernel = make_2dgaussian_kernel(fwhm=3.0, size=5)
    conv_hdu = convolve(data, kernel)
    seg_map = detect_sources(conv_hdu, threshold, npixels=9, mask=mask1) #1차 천체 탐지
    segm_deblend = deblend_sources(conv_hdu, seg_map,
                               npixels=30, nlevels=32, contrast=0.0005,
                               progress_bar=False) #천체분리

    segm_d = np.array(segm_deblend).astype(np.int32)
    x1,y1 = hdu.shape
    x, y = int(x1/2), int(y1/2)
    arr = segm_d[1380:1650,1380:1650]#[2400:3200,3200:4000] #[1050:2000,1200:2000] #중앙부 크롭
    seg_img = SegmentationImage(arr)
    
    #plt.imshow(seg_img, origin='lower'); plt.show(); sys.exit()
    
    labels = [x for x in seg_img.labels if x>=1208] #M 101에 해당하는 값을 가진 label의 리스트
    seg_img.remove_labels(labels) #M 101의 segmentation 제거
    segm_d_crop = np.array(seg_img)
    segm_d[1380:1650,1380:1650] = segm_d_crop #합성 #[1050:2000,1200:2000] #[x-400:x+200, y-300:y+400]

    segm = SegmentationImage(segm_d)
    cat = SourceCatalog(segm, segm, convolved_data=conv_hdu)
     
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
    #kernel0 = disk(3) 
    #seg_d = binary_dilation(segm_d, kernel0, iterations=1) #ngrow
    masked_map = np.where(segm_d!=0, 1, 0) + arr_zero #region 마스크 영상과 segmentation 마스크 영상을 합침
    masked = np.where(masked_map!=0, 1, 0).astype(np.int8)
    #masked[1610:1855,1487:1868] = 1 #NGC 5194
    #masked[1297:1598,1486:1821] = 1 #NGC 5195
    
    return np.array(masked, dtype=np.int8)

import warnings

warnings.filterwarnings('ignore')
hdu = fits.open('/volumes/ssd/intern/25_summer/NGC4236_r/sky_subed/coadd.fits')[0].data
#mask = fits.open('/volumes/ssd/intern/25_summer/M101_L/mask_coadd.fits')[0].data
#x,y = hdu.shape
mask = region_mask(hdu,3.0)
#plt.imshow(mask, origin='lower')
map = np.where(mask!=0, np.nan, hdu)
#fits.writeto('~/data/obj_rejec_IC3280_r.fits', mask, overwrite=True)
#map1 = np.where(map==0,np.nan, map)
#plt.imshow(mask, origin='lower')
#plt.imshow(hdu, origin='lower') #vmax=median+3*std, vmin=median-3*std,
median = np.nanmedian(map)
std = np.nanstd(map)
plt.imshow(map,vmax=median+3*std, vmin=median-3*std ,origin='lower')
#plt.imshow(map[int(x/2-1300):int(x/2+1300),int(y/2-1300):int(y/2+1300)],vmax=median+3*std, vmin=median-3*std,
            #origin='lower')
#plt.colorbar()
plt.show()
