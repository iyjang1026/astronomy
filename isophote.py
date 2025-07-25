import numpy as np
import sys
import astropy.io.fits as fits
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.wcs import WCS
from astropy.convolution import convolve
from photutils.segmentation import SegmentationImage, detect_sources, deblend_sources ,make_2dgaussian_kernel, SourceCatalog
from photutils.background import MedianBackground, Background2D
from photutils.isophote import EllipseGeometry, Ellipse, build_ellipse_model, IsophoteList, Isophote
import matplotlib.pyplot as plt
hdu = fits.open('/volumes/ssd/intern/25_summer/M101_L/sky_subed/coadd.fits')[0].data 
mask = fits.open('/volumes/ssd/intern/25_summer/M101_L/obj_rejec_coadd.fits')[0].data 
def detect(hdu0, mask):
    mask1 = np.where(hdu0!=0, False, True) #테두리 부분 0 마스크
    hdu = np.where(mask!=0, 0, hdu0) #웝본 데이터
    mean, median, std = sigma_clipped_stats(hdu, cenfunc='median', stdfunc='mad_std', sigma=3)
    arr_max_st = np.where(hdu>median+5*std, median+5*std, hdu)
    arr_st = np.where(arr_max_st<median-5*std, median-5*std, arr_max_st) #영상 stretch
    bkg_est = MedianBackground()
    bkg = Background2D(arr_st, (64,64), filter_size=(3,3), bkg_estimator=bkg_est, mask=mask+mask1) 
    data = hdu - bkg.background #background 제거
    threshold = 1.5*bkg.background_rms
    kernel = make_2dgaussian_kernel(fwhm=3.0, size=5)
    conv_hdu = convolve(data, kernel)
    seg_map = detect_sources(conv_hdu, threshold, npixels=5) #1차 천체 탐지
    
    segm_deblend = deblend_sources(conv_hdu, seg_map,
                               npixels=5000, nlevels=32, contrast=0.001,
                               progress_bar=False) #중심부 M 101 탐지

    cat = SourceCatalog(hdu, segm_deblend,convolved_data=conv_hdu)
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
        if eps > 0.8:
            a_list.append(0)
        else:
            a_list.append(b)
    max_idx = np.argmax(a_list) #단반경이 가장 긴 타원
    obj = ap[max_idx]
    eps = np.sqrt(1-(obj.b / obj.a)**2)
    x,y = obj.positions
 
    geometry = EllipseGeometry(x0=x, y0=y, sma=0.8*obj.a, eps=eps, pa=np.array(obj.theta)) 
    conv = convolve(hdu, kernel=kernel)
    ellipse = Ellipse(conv, geometry)
    isolist = ellipse.fit_image(fix_center=True, fflag=0.5) #isophote
    model = build_ellipse_model(hdu.shape, isolist) #modeling
    return model, isolist

model, isolist = detect(hdu, mask)
sma_arcsec = isolist.sma * 1.89
d = 6730 #M 101까지의 거리
kpc = d * np.tan((np.pi/180)*(sma_arcsec/3600))
fig, axes = plt.subplots(1,2)
fits.writeto('/volumes/ssd/intern/25_summer/M101_L/model.fits', model)
axes[0].imshow(np.sqrt(model), origin='lower')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('Model')
axes[1].plot(kpc, np.log10(isolist.intens), '.-')
axes[1].set_xlabel('Semi-Major Axis(kpc)')
axes[1].set_ylabel('Log10 Level')
axes[1].set_title('Radial Profile')
plt.show()
