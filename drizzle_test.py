from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord
from drizzle.resample import Drizzle
from drizzle.utils import calc_pixmap
import time
import glob
import sys 
import matplotlib.pyplot as plt
import numpy as np
import warnings
import progressbar

warnings.filterwarnings('ignore')
path= "/volumes/ssd/intern/25_summer/M101_L/sky_subed"

def driz(path):
    start_time = time.time()
    hdl_list = sorted(glob.glob(path+'/solv_*.fits'))
    bar1 = progressbar.ProgressBar(maxval=len(hdl_list), widgets=['[',progressbar.Timer(),']',progressbar.Bar()]).start()
    cen_sum_l = []
    for k in hdl_list:
        hdr = fits.open(k)[0].header
        wcs = WCS(hdr)
        cent_coord = wcs.pixel_to_world_values(2048,2048)
        cent_sum = np.sum(cent_coord)
        cen_sum_l.append(cent_sum)
    idx = np.argmax(np.array(cen_sum_l))
    ref_wcs = WCS(fits.open(hdl_list[idx])[0].header)

    d = Drizzle(out_shape=(4000,4000))

    for i in range(5):
        hdl = fits.open(hdl_list[i])
        hdu = hdl[0].data
        hdr = hdl[0].header
        hdl.close()
        wcs = WCS(hdr)   
        wht_map = ~np.isnan(hdu) 
        pixmap=calc_pixmap(wcs, ref_wcs)

        d.add_image(hdu, exptime=30, pixmap=pixmap, scale=1.0, weight_map=wht_map,
                    xmin=-476, xmax=2048+476, ymin=-476, ymax=2048+476)
        bar1.update(i)
    bar1.finish()
    end_time = time.time()
    eta = end_time - start_time
    print(f'{eta//60} min {eta-(eta//60)*60} seconds')
    output = d.out_img
    ctx = d.ctx_id
    print(ctx)
    idx = np.where(np.array(~np.isnan(output))==1)
    out = output[:np.max(idx[0]),:np.max(idx[1])]
    mean, median, std = sigma_clipped_stats(out, cenfunc='median', stdfunc='mad_std')
    #fits.writeto(path+'/coadd_driz.fits',out,overwrite=True)
    plt.imshow(out,vmax=median+3*std, vmin=median-3*std, origin='lower') #vmax=median+3*std, vmin=median-3*std,
    plt.show()

driz(path)

