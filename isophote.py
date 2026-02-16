import numpy as np
import sys
import astropy.io.fits as fits
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.wcs import WCS
from astropy.convolution import convolve
import astropy.units as u
from photutils.segmentation import SegmentationImage, detect_sources, deblend_sources ,make_2dgaussian_kernel, SourceCatalog
from photutils.background import MedianBackground, Background2D
from photutils.isophote import EllipseGeometry, Ellipse, build_ellipse_model, IsophoteList, Isophote
from photutils.aperture import EllipticalAperture
import matplotlib.pyplot as plt
import os

def detect(hdu0, mask,i, eps_filter):
    #hdu = np.ma.masked_where(mask|(hdu0>1500),np.ma.masked_equal(hdu0, np.zeros(shape=hdu0.shape)))
    hdu = np.ma.masked_where(mask,np.ma.masked_equal(hdu0, np.zeros(shape=hdu0.shape))-i)
    #plt.imshow(hdu, origin='lower');plt.show();sys.exit()
    #hdu = hdu1 +abs(np.min(hdu0))#3*np.ma.std(hdu1) #테두리와 region 마스크를 적용
    bkg_est = MedianBackground()
    bkg = Background2D(hdu, (64,64), filter_size=(3,3), bkg_estimator=bkg_est) #배경추출
    data = hdu - bkg.background #background 제거
    threshold = 3.0*bkg.background_rms #threshold 설정
    kernel = make_2dgaussian_kernel(fwhm=3.0, size=5)
    conv_hdu = convolve(data, kernel)
    seg_map = detect_sources(conv_hdu, threshold, npixels=5) #1차 천체 탐지
    
    segm_deblend = deblend_sources(conv_hdu, seg_map,
                               npixels=2000, nlevels=32, contrast=0.0005,
                               progress_bar=False) #중심부 M 101 탐지
    #plt.imshow(segm_deblend, origin='lower'); plt.show(); sys.exit()
    """
    #중심부 crop
    x1, y1 = hdu.shape
    x, y = int(x1/2), int(y1/2)
    seg_arr = np.array(segm_deblend)[x-300:x+150, y-200:y+100]#[1600:1700, 1540:1640]#
    #plt.imshow(seg_arr, origin='lower'); plt.show(); sys.exit()
    segm = SegmentationImage(seg_arr)
    cat = SourceCatalog(hdu[x-300:x+150, y-200:y+100], segm,convolved_data=conv_hdu[x-300:x+150, y-200:y+100]) #천체 카탈로그 생성
    """
    cat = SourceCatalog(hdu, segm_deblend, convolved_data=conv_hdu)
    """
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
        if eps_filter == True:
            if eps > 0.8: #특정 이심률 필터링
                a_list.append(0)
            else:
                a_list.append(b)
        else:
            a_list.append(a)
    """
    a_list = list(cat.semiminor_sigma.value)
    max_idx = np.argmax(a_list) #단반경이 가장 긴 타원의 인덱스 반환
    obj = cat[max_idx]

    eps = np.sqrt(1-(obj.semiminor_sigma.value / obj.semimajor_sigma.value)**2)
    aper = EllipticalAperture((obj.xcentroid, obj.ycentroid), obj.semimajor_sigma.value*3, obj.semiminor_sigma.value*3, obj.orientation.value*np.pi/180)
    #x_p,y_p = obj.xcentroid, obj.ycentroid #obj.positions
    x,y = obj.xcentroid, obj.ycentroid #obj.positions
    #print(obj.orientation);sys.exit()
    #geometry = EllipseGeometry(x0=x_p+(x-200), y0=y_p+(y-300), sma=obj.a, eps=eps, pa=np.array(obj.theta)) #isophote를 위한 초기 타원 생성
    geometry = EllipseGeometry(x0=x, y0=y, sma=obj.semimajor_sigma.value*3, eps=obj.ellipticity.value, pa=obj.orientation.value*np.pi/180) #isophote를 위한 초기 타원 생성
    #plt.imshow(segm_deblend, origin='lower'); aper.plot(color='C3'); plt.show(); sys.exit()
    return geometry, obj.semimajor_sigma.value*3
    #hdu1 = np.ma.masked_where(hdu>40000, hdu)
    
def ellipse(hdu, geometry, sma):    
    ellipse = Ellipse(hdu, geometry)
    isolist = ellipse.fit_image(sma0=0.8*sma,maxsma=2.*sma, integrmode='median',step=0.05, sclip=3.0, nclip=3, fflag=0.3, fix_center=True) #isophote #minsma=0.05*sma,
    tbl = isolist.to_table()    
    print(tbl)
    #fill = np.median(hdu[1050:2000,1200:2000])
    
    model = build_ellipse_model(hdu.shape, isolist) #modeling
    
    return model, tbl#, np.ma.std(hdu)
    
    #return tbl

path = '/Users/jang-in-yeong/data/ic3280'
color = 'g'
obj = 'IC3280'
hdu = fits.open(path+'/'+obj+'_'+color+'.fits')[0].data
mask = fits.open(path+'/obj_rejec_'+obj+'_'+color+'.fits')[0].data 

"""
mask1 = fits.open(path+'/mask_'+obj+'_'+color+'.fits')[0].data
from SBLimit import bkg_std
std_arr, median_arr = bkg_std(hdu, mask1, 128)
mena, median, sigma = sigma_clipped_stats(median_arr, cenfunc='median', stdfunc='mad_std', sigma=3)

path = "/volumes/ssd/intern/25_summer/NGC4236_r"
hdu = fits.open(path+'/sky_subed/coadd.fits')[0].data
mask = fits.open(path+'/obj_rejec_coadd.fits')[0].data 
color = 'r'
"""
fig, ax = plt.subplots(1,3, figsize=(15,5), gridspec_kw={'width_ratios':[1,1,1]})
"""
if not os.path.exists(path+'/tbl_median'):
    os.mkdir(path+'/tbl_median')
"""


geo, sma0 = detect(hdu, mask,0, eps_filter=False)

model,tbl = ellipse(hdu, geo, sma0)
fits.writeto(path + '/model_'+color+'.fits', model, overwrite=True)
#tbl = ellipse(hdu, geo, sma0)
radius = tbl['sma'] * 1.86
intens = tbl['intens'] / (1.86**2)
mag = -2.5*np.log10(intens)+28.51
tbl.write(path+'/test_color_iso_tbl_'+color+'.csv', format='ascii.csv', overwrite=True)
ax[0].scatter(radius, mag, s=2)
ax[1].imshow(np.log10(model), origin='lower')
ax[2].imshow(np.log10(hdu), origin='lower')

"""
#mean, median, std = sigma_clipped_stats(hdu, cenfunc='median', stdfunc='mad_std', sigma=3)
for i in np.linspace(median-3*sigma, median+3*sigma, 3):
    print(i)
    
    hdu0 = hdu - i
    tbl = ellipse(hdu0, geo, sma0)
    tbl.write(path+'/tbl_median/tbl_'+str(-i)+'.csv', format='ascii.csv', overwrite=True)
    sma = tbl['sma']*0.262
    d = 8580 #M 101까지의 거리(kpc)
    kpc = d * np.tan((np.pi/180)*((sma)/3600))
    intens = tbl['intens']/(1.89**2)
    mag = -2.5*np.log10(intens)+28.44
    ax.plot(kpc, mag, label=f'hdu-{i}')
"""
#sys.exit()
ax[0].set_xlabel('sma(arcsec)')
ax[0].set_ylabel('$\mu_r$')
ax[0].invert_yaxis()
plt.legend()
plt.show()

"""
geo, sma0 = detect(hdu, mask,0) #
tbl = ellipse(hdu, geo, sma0)
#tbl.write(path+'/tbl/t_iso_tbl.csv', format='ascii.csv', overwrite=True)
#sys.exit()
#print(std)
sma_arcsec = tbl['sma'] * 1.89 #pixel 단위에서 arcsec 단위로 변경
d = 8580 #M 101까지의 거리(kpc)
kpc = d * np.tan((np.pi/180)*(sma_arcsec/3600)) #arcsec 단위에서 kpc단위로 변경

fig, axes = plt.subplots(2,2)
#fits.writeto('/volumes/ssd/intern/25_summer/NGC5907_r/model_test.fits', model, overwrite=True)

m = axes[0,0].imshow(model, origin='lower')
plt.colorbar(m, ax=axes[0,0])
axes[0,0].set_xlabel('x')
axes[0,0].set_ylabel('y')
axes[0,0].set_title('Model')

axes[0,1].plot(kpc, -2.5*np.log10((tbl['intens'])/(1.89**2))+28.44, '.-')
#axes[0,1].set_xlim(-1,32)
axes[0,1].invert_yaxis()
axes[0,1].set_xlabel('Semi-Major Axis(kpc)')
axes[0,1].set_ylabel(f'$Mag_r$')
axes[0,1].set_title('Radial Profile')

axes[1,0].plot(kpc, tbl['pa']/np.pi*180, '.-')
axes[1,0].set_xlabel('Semi-Major Axis(kpc)')
axes[1,0].set_ylabel('PA(deg)')
axes[1,0].set_title('PA Profile')

axes[1,1].plot(kpc, tbl['ellipticity'],'.-')
axes[1,1].set_xlabel('Semi-Major Axis(kpc)')
axes[1,1].set_ylabel('eps')
axes[1,1].set_title('Ellipticity Profile')
plt.show()
"""