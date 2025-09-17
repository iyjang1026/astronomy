import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.convolution import convolve
from photutils.segmentation import detect_sources, SourceCatalog, deblend_sources
from photutils.background import Background2D, MedianBackground
from photutils.aperture import EllipticalAperture
import sys
import glob
import ray

path = "~/Downloads/Light_east_30.0s_Bin1_20250605-032507_0738.fit"

def satelite(path):
    hdu = fits.open(path)[0].data
    mean, median, std = sigma_clipped_stats(hdu, cenfunc='median', stdfunc='mad_std')
    high = np.where(median+3*std<=hdu, median+3*std, hdu)
    arr = np.where(median-3*std>=high, median-3*std, high)
    bkg_est = MedianBackground()
    bkg = Background2D(arr, (64,64), filter_size=(5,5), bkg_estimator=bkg_est)
    data = arr - bkg.background
    thr = 1.5*bkg.background_rms
    kernel = np.array([[0,1,0],[1,1,1],[0,1,0]])
    conv_hdu = convolve(data, kernel)
    seg = detect_sources(conv_hdu, thr, npixels=5)
    cat = SourceCatalog(data, seg, convolved_data=conv_hdu)
    cat_idx = np.where(cat.ellipticity>0.9)[0]
    #print(len(cat_idx));sys.exit()
    sma_l = list(cat.semimajor_sigma.value.astype(np.float32))
    """
    tmp = sma_l.copy()
    tmp.sort()
    tmp_num = tmp[-200:]
    top_idx = [sma_l.index(x) for x in tmp_num]
    """
    pa = []
    aper_l = []
    for i in cat_idx:
        cati = cat[i]
        x,y = cati.xcentroid, cati.ycentroid
        maj, min = cati.semimajor_sigma.value*3, cati.semiminor_sigma.value*3
        theta = cati.orientation.value * np.pi/180   
        aper = EllipticalAperture((x,y), maj, min, theta)
        #aper.plot(color='C3')
        aper_l.append(aper)
        pa.append(cati.orientation.value)
    pa = np.array(pa)
    mean, median, std = sigma_clipped_stats(pa, cenfunc='median', stdfunc='mad_std')
    aper_l = np.array(aper_l)
    upper, lower = median+4*std, median-4*std
    out_diff = np.where((pa>=upper)|(pa<=lower), pa, np.nan)
    masked_arr = np.ma.masked_invalid(out_diff)
    idx = np.where(masked_arr.mask!=True)[0].astype(np.int16)
    apers = aper_l[idx]

    #print(type(path));sys.exit()
    #plt.imshow(data, origin='lower')
    """
    for i in apers:
        i.plot(color='C3')
    """
    #plt.show()
    if len(apers)!=0:
        return path, len(apers)
    else:
        return None
    
    
    
import cv2
def line_detect(path):
    hdu = fits.open(path)[0].data
    edges = cv2.Canny(np.array(hdu).astype(np.uint8),500,150)
    
    lines = cv2.HoughLines(edges, 0.8, np.pi/180,100, min_theta=0, max_theta=2*np.pi)
    print(lines)
    plt.imshow(edges, origin='lower')
    plt.show()

if __name__=="__main__":
    
    import warnings
    warnings.filterwarnings('ignore')
    file_txt = open('/volumes/ssd/BSH_data/250820/sate_0.txt', 'a')
    file = sorted(glob.glob('/volumes/ssd/BSH_data/250820/0/bin2*.fits'))
    for i in range(len(file)):
        d_type = satelite(file[i])
        
        if d_type != None:
            img_path, num = d_type
            file_txt.write(f'{img_path}, {num}\n')
        
        print(f'complete {i}')
 
    file_txt.close()
   
    #satelite(path)
    #line_detect(path)