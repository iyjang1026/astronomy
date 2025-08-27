import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, Column
from astropy.modeling.models import Sersic1D
import sys
import glob
from scipy.optimize import curve_fit

d = 6730
def kpc(x):
    kpc = d * np.tan((np.pi/180)*((x*1.89)/3600)) #장반경을 pixel 단위에서 kpc으로 변환
    return kpc

def sersic(x,amp,r_eff,n): #서직윤곽 함수
    de_v_r = Sersic1D(r_eff=r_eff, amplitude=amp, n=n)(x)
    return de_v_r

def mag(x): #등급으로 변환
    mag = -2.5*np.log10(x)+29.04 #28.05 #29.04 #28.39
    return mag

def iso_err_plot(path):
    iso_list = glob.glob(path+'/tbl_median_test_1/*.csv')
    print(iso_list)
    err_l = []
    num_list = []
    #fig, ax = plt.subplots()
    for i in range(len(iso_list)):
        tbl = Table.read(iso_list[i], format='ascii.csv')
        intens = tbl['intens']/(1.89**2)
        list = [np.nan if x<=0 else x for x in intens]
        #num_list.append(len(intens))
        err_l.append(list)
    
    
    cut_idx = len(err_l[0])#num_list[np.argmax(num_list)]#
    """
    for j in range(len(iso_list)):
        tbl = Table.read(iso_list[j], format='ascii.csv')
        intens = tbl['intens']/(1.89**2)
        list = [np.nan if x<=0 else x for x in intens][:cut_idx]
        for k in range(cut_idx-len(list)):
            list.append(np.nan)
        err_l.append(list)
    """
    err_arr = np.array(err_l)#[:,:-1]
    
    max_idx = np.nanargmax(err_arr, axis=0)
    min_idx = np.nanargmin(err_arr, axis=0)
    median_arr = np.nanmedian(err_arr, axis=0)
    max_arr = np.zeros((len(max_idx),))
    min_arr = np.zeros((len(min_idx),))
    for m in range(len(max_idx)):
        max_arr[m] += err_arr[max_idx[m],m]
    for min in range(len(min_idx)):
        min_arr[min] += err_arr[min_idx[min],min]
    #print(max_arr-min_arr)
    min_err = np.where(min_arr<0, -2.5*np.log10(abs(min_arr)), 2.5*np.log10(min_arr))
    max_err = 2.5*np.log10(max_arr)
    #print(cut_idx)
    return max_err - min_err, max_arr, cut_idx, median_arr

#iso_err_plot('/volumes/ssd/intern/25_summer/NGC6946_L')
"""
path = '/volumes/ssd/intern/25_summer/M101_L'
tbl = Table.read(path+'/tbl/t_iso_tbl.csv', format='ascii.csv')
sma = tbl['sma']*1.89
intens = tbl['intens']/(1.89**2)
intens_err = tbl['intens_err']/(1.89**2)
fig, ax = plt.subplots(figsize=(5,5))
#ax.scatter(kpc, mag(intens), label='data', s=3, c='grey')
err, max_err, cut_idx, median_arr = iso_err_plot(path)
ax.plot(sma[:cut_idx],mag(median_arr), label='original')
#ax.plot(sma, (mag(max_err)+err))
#print(mag(max_err))
ax.fill_between(sma[:cut_idx], mag(max_err), mag(max_err)+err, color='skyblue') #2.5*np.log10(intens_err[:cut_idx])/2
#ax.errorbar(sma, mag(intens), yerr=err/2)
ax.set_xlabel('sma(arcsec)')
ax.set_ylabel('$Mag_r$')
#ax.set_ylim(17,30)
ax.invert_yaxis()
#ax.set_yscale('log', base=10)
ax.set_title('Profile')
#plt.xscale('log', base=10)
#plt.yscale('log', base=10)
plt.legend()
plt.show()
"""
"""
        ax.plot(kpc(sma), mag(intens), label=str(i))
        ax.invert_yaxis()
    plt.legend()
    plt.show()
iso_err_plot('/volumes/ssd/intern/25_summer/M51_L')
    """