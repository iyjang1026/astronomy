import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.coordinates as coord
import astropy.units as u
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from astropy.stats import sigma_clipped_stats, sigma_clip
import warnings
import sys
from scipy.stats import mode


warnings.filterwarnings('ignore')
def se_mask(file):
    import sep
    data = fits.open(file)[0].data
    hdr = fits.open(file)[0].header
    data = np.array(data, dtype=float)
    data1 = data.astype(data.dtype.newbyteorder('='))
    bkg = sep.Background(data1)
    subd = data - bkg
    obj, seg_map = sep.extract(subd, bkg.globalrms, segmentation_map=True)
    mask_map = np.array(seg_map)
    from scipy.ndimage import binary_dilation
    kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])
    mask_map_d = binary_dilation(mask_map, kernel, iterations=2)
    masked = np.where((mask_map_d!=0), np.nan, data)
    mean, median, std = sigma_clipped_stats(masked, cenfunc='median', stdfunc='mad_std', sigma=3)
    print(std)
    return std

def bkg_std(hdu, mask, size):
    std_list = []
    median_list = []
    arr = np.ma.masked_where(mask, np.ma.masked_equal(hdu, 0))#np.where(mask!=0, np.nan, hdu) #
    """
    mean, median, std = sigma_clipped_stats(arr, cenfunc='median', stdfunc='mad_std', sigma=3.)
    return std
    """
    #plt.imshow(arr1, origin='lower'); plt.show(); sys.exit()
    x,y = arr.shape
    center_x, center_y = int(x/2), int(y/2)
    #bin_arr = arr[center_x-1250:center_x+1250, center_y-1250:center_y+1250]
    for i in range(1000):
        rand_st_x, rand_st_y = np.random.randint(center_x-1500,center_x+1500-size, 2)
        bin_x, bin_y = rand_st_x+size, rand_st_y+size
        bin_arr = arr[rand_st_x:rand_st_x+size, rand_st_y:rand_st_y+size]
        mean, median, std = sigma_clipped_stats(bin_arr, cenfunc='median', stdfunc='mad_std', sigma=3)
        #std = np.ma.std(bin_arr)
        median_list.append(median)
        std_list.append(std)
    mean, std_median, std = sigma_clipped_stats(np.array(std_list).astype(np.float32),
                                            cenfunc='median', stdfunc='mad_std', sigma=3.)
    
    return std_median, np.array(median_list, dtype=np.float32)

def hist(hdu, mask):
    data = np.ma.masked_where(mask, np.ma.masked_equal(hdu, 0))
    mean, median, std = sigma_clipped_stats(data, cenfunc='median', stdfunc='mad_std', sigma=3)
    print(std)
    hist, bin = np.histogram(data, bins=256)
    plt.hist(hist, bin, label='image_hist')
std_hdu = fits.open('/volumes/ssd/intern/25_summer/M101_L/sky_subed/coadd.fits')[0].data 
std_mask = fits.open('/volumes/ssd/intern/25_summer/M101_L/mask_coadd.fits')[0].data 
std_noise, median_arr = bkg_std(std_hdu, std_mask, 128)

def median_std(hdu, mask, iter):
    median_list = []
    for i in range(iter):
        std_noise, median_arr = bkg_std(hdu, mask, 128)
        median_list.append(median_arr)
    mean, median, median_std = sigma_clipped_stats(np.array(median_list), cenfunc='median', stdfunc='mad_std', sigma=3.)

    print(f'std is {median_std, median}')
#median_std(std_hdu, std_mask,50)
#hist(std_hdu, std_mask)
#plt.hist(median_arr, bins=256, label='median_hist'); plt.show()

def sb_limit():
    #read catalogue
    source = '/volumes/ssd/2025-09-15/l/coadd.cat'#
    data = Table.read(source, format='ascii', converters={'obsid':str})
    #check the catalogue location
    sdss = Table.read('~/M27_GSC.csv', format='ascii') #check!! 

    #extract coordinate
    sdsscat = sdss['RA_ICRS', 'DE_ICRS', 'gmag','rmag']
    objcat = data['ALPHAPEAK_J2000','DELTAPEAK_J2000','FLUX_BEST', 'ERRAWIN_IMAGE', 'ERRBWIN_IMAGE']
    #obj_cat = objcat[(objcat['ERRAWIN_IMAGE']<0.01)&(objcat['ERRBWIN_IMAGE']<0.01)]
    sdss_coord = SkyCoord(ra=sdsscat['RA_ICRS']*u.degree, dec=sdsscat['DE_ICRS']*u.degree, frame='fk5')
    obj_coord = SkyCoord(ra=objcat['ALPHAPEAK_J2000'], dec=objcat['DELTAPEAK_J2000'], frame='fk5')


    idx1, d2d1, d3d1 = sdss_coord.match_to_catalog_sky(obj_coord)
    sdss_data = sdsscat
    obj = objcat[idx1]


    obj_flux = obj['FLUX_BEST']
    sdss_mag = sdss_data['rmag']
    count = np.array(obj_flux)
    mag = np.array(sdss_mag)


    from astropy.stats import sigma_clipped_stats

    m = -2.5*np.log10(count)


    mM = mag - m


    from scipy.optimize import curve_fit

    def log(x,a,c):
        return a*np.log10(x)+c 
    
    mean, median1, std = sigma_clipped_stats(mM, cenfunc='median', stdfunc='mad_std', sigma=3.0)
    z = np.where((mM<=median1+3*std)&(mM>=median1-3*std), mM, np.nan) # sigma_clip(mM, cenfunc='median', stdfunc='mad_std', sigma=3) #
    mag_ob = m + z

    t_r = count[~np.isnan(mag_ob)]
    popt, pcov = curve_fit(log, t_r, mag_ob[~np.isnan(mag_ob)])
    print(popt[0])
    print(f'Z_p is {popt[1]}')
    print(f'SB Limit is {popt[1] - 2.5*np.log10(std_noise/(1.89*10))}')
    
    plt.scatter(count, mag, s=2, c='grey')
    plt.scatter(count, mag_ob, s=2, c='r')
    count.sort()
    plt.plot(count, log(count, *popt), linewidth=1.5, c='k')
    plt.xscale('log', base=10)
    #plt.xlim(10**3.35,10**6.1)
    plt.xlabel('Flux(log10)')
    plt.ylabel('$Mag_{SDSS,r}$')
    sb_lim = popt[1] - 2.5*np.log10(std_noise/(1.89*10))
    plt.text(10**3.5, 10, f'$Z_p$ = {popt[1]:.2f}\nSB Limit = {sb_lim:.2f}', bbox={'boxstyle':'square', 'fc':'white'})
    plt.title('M27')

    plt.show()

sb_limit()

"""
for i in range(10):
    print(f'{i+1} -------------------')
    std_noise = bkg_std(std_hdu, std_mask, 128)
    print(f'std is {std_noise}')
    sb_limit()
"""