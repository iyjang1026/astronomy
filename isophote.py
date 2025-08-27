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
import os

def detect(hdu0, mask,i, eps_filter):
    hdu = np.ma.masked_where(mask,np.ma.masked_equal(hdu0, np.zeros(shape=hdu0.shape)-i))
    #hdu = hdu1 +abs(np.min(hdu0))#3*np.ma.std(hdu1) #테두리와 region 마스크를 적용
    bkg_est = MedianBackground()
    bkg = Background2D(hdu, (64,64), filter_size=(3,3), bkg_estimator=bkg_est) #배경추출
    data = hdu - bkg.background #background 제거
    threshold = 3.0*bkg.background_rms #threshold 설정
    kernel = make_2dgaussian_kernel(fwhm=3.0, size=5)
    conv_hdu = convolve(data, kernel)
    seg_map = detect_sources(conv_hdu, threshold, npixels=5) #1차 천체 탐지
    
    segm_deblend = deblend_sources(conv_hdu, seg_map,
                               npixels=80, nlevels=32, contrast=0.0005,
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
    max_idx = np.argmax(a_list) #단반경이 가장 긴 타원의 인덱스 반환
    obj = l[max_idx]

    eps = np.sqrt(1-(obj.b / obj.a)**2)

    x_p,y_p = obj.positions
    x,y = obj.positions
    #geometry = EllipseGeometry(x0=x_p+(x-200), y0=y_p+(y-300), sma=obj.a, eps=eps, pa=np.array(obj.theta)) #isophote를 위한 초기 타원 생성
    geometry = EllipseGeometry(x0=x, y0=y, sma=obj.a, eps=eps, pa=np.array(obj.theta)) #isophote를 위한 초기 타원 생성
    #plt.imshow(segm_deblend, origin='lower'); obj.plot(color='C3'); plt.show(); sys.exit()
    return geometry, obj.a
    #hdu1 = np.ma.masked_where(hdu>40000, hdu)
    
def ellipse(hdu, geometry, sma):    
    ellipse = Ellipse(hdu, geometry)
    isolist = ellipse.fit_image(sma0=0.3*sma,maxsma=114.30/1.89,integrmode='bilinear',step=0.07, sclip=3.0, nclip=3, fflag=0.3, fix_center=True, fix_pa=False, fix_eps=False) #isophote
    tbl = isolist.to_table()    
    print(tbl)
    #fill = np.median(hdu[1050:2000,1200:2000])
    #model = build_ellipse_model(hdu.shape, isolist) #modeling

    #return model, tbl, np.ma.std(hdu)
    return tbl

path = '/volumes/ssd/intern/25_summer/M51_L'
hdu = fits.open(path + '/sky_subed/coadd.fits')[0].data 
mask = fits.open(path+'/M51b_obj_rejec_coadd.fits')[0].data 
mask1 = fits.open(path+'/mask_coadd.fits')[0].data
from SBLimit import bkg_std
std_arr, median_arr = bkg_std(hdu, mask1, 128)
mena, median, sigma = sigma_clipped_stats(median_arr, cenfunc='median', stdfunc='mad_std', sigma=3)

fig, ax = plt.subplots()
if not os.path.exists(path + '/tbl_median_test_1'):
    os.mkdir(path + '/tbl_median_test_1')
geo, sma0 = detect(hdu, mask,0, eps_filter=True)
mean, median, std = sigma_clipped_stats(hdu, cenfunc='median', stdfunc='mad_std', sigma=3)
for i in np.linspace(round(median-3*sigma), round(median+3*sigma), 5):
    #print(i)
    
    hdu0 = hdu - i
    tbl = ellipse(hdu0, geo, sma0)
    tbl.write(path+'/tbl_median_test_1/tbl_'+str(-i)+'.csv', format='ascii.csv', overwrite=True)
    sma = tbl['sma']
    d = 8580 #M 101까지의 거리(kpc)
    kpc = d * np.tan((np.pi/180)*((sma*1.89)/3600))
    intens = tbl['intens']/(1.89**2)
    mag = -2.5*np.log10(intens)+28.44
    ax.plot(kpc, mag, label=f'hdu-{i}')
    
#sys.exit()
ax.set_xlabel('sma(kpc)')
ax.set_ylabel('$Mag_r$')
ax.invert_yaxis()
plt.legend()
plt.show()

"""
model, tbl, std = detect(hdu, mask,0) #
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