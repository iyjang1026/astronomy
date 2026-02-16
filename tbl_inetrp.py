from astropy.io import fits

def griz(path, obj_name):
    r = fits.open(path+'/'+obj_name+'_r.fits')[0].data
    g = fits.open(path + '/'+obj_name+'_g.fits')[0].data
    hdu = g / r 
    fits.writeto(path+'/'+obj_name+'_gr.fits', data=hdu, overwrite=True)

#griz('~/data/ic3280', 'IC3280')

from scipy import interpolate
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
import sys
import glob

path = '/Users/jang-in-yeong/data/ic3280'
pixel_scale = 0.262

def err_interp(path, pixelscale):
    tbl_r = Table.read(path+'/iso_tbl_r.csv', format='ascii')
    intens_r = tbl_r['intens'] / (pixelscale**2)
    r_err = tbl_r['intens_err']
    radius_r = tbl_r['sma'] * pixelscale
    #print(len(radius_r))
    out_tbl = [radius_r[:64],intens_r[:64],r_err]
    err_file = sorted(glob.glob(path+'/tbl_median/*csv'))
    #print(err_file);sys.exit()
    for i in err_file:
        table = Table.read(i,format='ascii')
        sma = table['sma'] * pixelscale
        intens = table['intens'] / (pixelscale**2)
        interp_intens = interpolate.interp1d(sma,intens, kind='linear')
        out_tbl.append(interp_intens(radius_r))
    tbl_f = Table(out_tbl,names=['sma','r','r_err','-1','0','1'])
    tbl_f.write(path+'/err.csv', format='ascii.csv',overwrite=True)
     
#err_interp(path, pixel_scale)

def interpol_tbl(path, pixel_scale):
    tbl_r = Table.read(path+'/test_color_iso_tbl_r.csv', format='ascii')
    tbl_g = Table.read(path+'/test_color_iso_tbl_g.csv', format='ascii')
    intens_r = tbl_r['intens'] / (pixel_scale**2)
    intens_g = tbl_g['intens'] / (pixel_scale**2)
    radius_r = tbl_r['sma'] * pixel_scale
    radius_g = tbl_g['sma'] * pixel_scale
    sma_r_max = np.max(radius_r)
    sma_g_max = np.max(radius_g)
    if sma_g_max >= sma_r_max:
        interp_g = interpolate.interp1d(radius_g, intens_g, kind='linear')
        interp_g_err = interpolate.interp1d(radius_g, tbl_g['intens_err'], kind='linear')
        tbl = Table([radius_r, interp_g(radius_r), interp_g_err(radius_r), intens_r,tbl_r['intens_err']], names=['sma','g','g_err','r','r_err'])
        tbl.write(path+'/test.csv', format='ascii.csv', overwrite=True)
        print(tbl)
        plt.plot(radius_r, intens_r,'.', label='r')
        plt.plot(radius_g, intens_g,'.', label='g')
        plt.plot(radius_r, interp_g(radius_r), label='g_interp')
        plt.plot(radius_r, interp_g(radius_r)/intens_r, label='g-r')
        plt.xlabel(f'sma(aracsec)')
        plt.ylabel(f'flux')
        plt.legend()
        plt.show()
        
    else:
        interp_r = interpolate.interp1d(radius_r, intens_r, kind='linear')
        interp_r_err = interpolate.interp1d(radius_r, tbl_r['intens_err'], kind='linear')
        tbl = Table([radius_g, intens_g, tbl_g['intens_err'], interp_r(radius_g), interp_r_err(radius_g)], names=['sma','g','g_err','r','r_err'])
        tbl.write(path+'/test.csv', format='ascii.csv', overwrite=True)
        print(tbl)
        plt.plot(radius_r, intens_r,'.', label='r')
        plt.plot(radius_g, interp_r(radius_g), label='r_interp')
        plt.plot(radius_g, intens_g/interp_r(radius_g), label='g-r')
        plt.xlabel(f'sma(aracsec)')
        plt.ylabel(f'flux')
        plt.legend()
        plt.show()

    
        
interpol_tbl(path, pixel_scale)
