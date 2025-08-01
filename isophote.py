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
    hdu = np.ma.masked_where(mask, np.ma.masked_equal(hdu0, np.zeros(shape=hdu0.shape)))
    #hdu = hdu1 +abs(np.min(hdu0))#3*np.ma.std(hdu1) #테두리와 region 마스크를 적용
    bkg_est = MedianBackground()
    bkg = Background2D(hdu, (64,64), filter_size=(3,3), bkg_estimator=bkg_est) #배경추출
    data = hdu - bkg.background #background 제거
    threshold = 1.5*bkg.background_rms #threshold 설정
    kernel = make_2dgaussian_kernel(fwhm=3.0, size=5)
    conv_hdu = convolve(data, kernel)
    seg_map = detect_sources(conv_hdu, threshold, npixels=5) #1차 천체 탐지
    
    segm_deblend = deblend_sources(conv_hdu, seg_map,
                               npixels=80, nlevels=32, contrast=0.0005,
                               progress_bar=False) #중심부 M 101 탐지

    cat = SourceCatalog(hdu, segm_deblend,convolved_data=conv_hdu) #천체 카탈로그 생성
    ap = cat.kron_aperture #천체에 맞는 타원 생성
    l = [x for x in ap if x!=None]
    a_list = []
    for i in l:
        a = None
        b = None
        eps = None
        a = i.a
        b = i.b
        eps = np.sqrt(1-(b/a)**2)
        if eps > 0.8: #특정 이심률 필터링
            a_list.append(0)
        else:
            a_list.append(b)
    
    max_idx = np.argmax(a_list) #단반경이 가장 긴 타원의 인덱스 반환
    obj = l[max_idx]
    
    eps = np.sqrt(1-(obj.b / obj.a)**2)
    x,y = obj.positions
 
    geometry = EllipseGeometry(x0=x, y0=y, sma=obj.a, eps=eps, pa=np.array(obj.theta)) #isophote를 위한 초기 타원 생성
    ellipse = Ellipse(hdu, geometry)
    isolist = ellipse.fit_image(integrmode='median', sclip=3.0, nclip=3, fflag=0.3, fix_center=True) #isophote
    tbl = isolist.to_table()
    tbl.write('/volumes/ssd/intern/25_summer/M101_L/iso_integ_median_tbl.csv', format='ascii.csv')
    #fill = np.median(hdu[1050:2000,1200:2000])
    model = build_ellipse_model(hdu.shape, isolist) #modeling
    return model, isolist, np.ma.std(hdu)

model, isolist, std = detect(hdu, mask) #
print(std)
sma_arcsec = isolist.sma * 1.89 #pixel 단위에서 arcsec 단위로 변경
d = 6730 #M 101까지의 거리(kpc)
kpc = d * np.tan((np.pi/180)*(sma_arcsec/3600)) #arcsec 단위에서 kpc단위로 변경
fig, axes = plt.subplots(2,2)
fits.writeto('/volumes/ssd/intern/25_summer/M101_L/model_integ_median.fits', model)
m = axes[0,0].imshow(model, origin='lower')
plt.colorbar(m, ax=axes[0,0])
axes[0,0].set_xlabel('x')
axes[0,0].set_ylabel('y')
axes[0,0].set_title('Model')

axes[0,1].plot(kpc, -2.5*np.log10((isolist.intens)/(1.89**2))+28.39, '.-')
#axes[0,1].set_xlim(-1,32)
axes[0,1].invert_yaxis()
axes[0,1].set_xlabel('Semi-Major Axis(kpc)')
axes[0,1].set_ylabel(f'$Mag_r$')
axes[0,1].set_title('Radial Profile')

axes[1,0].plot(kpc, isolist.pa/np.pi*180, '.-')
axes[1,0].set_xlabel('Semi-Major Axis(kpc)')
axes[1,0].set_ylabel('PA(deg)')
axes[1,0].set_title('PA Profile')

axes[1,1].plot(kpc, isolist.eps,'.-')
axes[1,1].set_xlabel('Semi-Major Axis(kpc)')
axes[1,1].set_ylabel('eps')
axes[1,1].set_title('Ellipticity Profile')
plt.show()
