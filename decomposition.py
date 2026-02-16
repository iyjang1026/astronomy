import numpy as np
from astropy.table import Table
from astropy.modeling.models import Sersic1D
from astropy.stats import sigma_clipped_stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from isolist_plot import iso_err_plot
import sys
import warnings

warnings.filterwarnings('ignore')

def imp_tbl(path):
    tbl = Table.read(path, format='ascii.csv')
    return tbl

def sersic(x,amp, r_eff, n):
    return Sersic1D.evaluate(x, amp,r_eff, n)

def exponential(x, amp, r_eff):
    return amp*np.exp(-(x/r_eff))

def mag(x, z_p):
    return -2.5*np.log10(x)+z_p

def arcsec(x):
    return (x*1.89)#
def kpc(x,d):
    return d * np.tan((np.pi/180)*((x)/3600))

def step_func(x, d):
    step = [x<d]
    return step

def sum_profile(x,a1,r1,a2,r2,n,r_s):
    #w1 = x<=d
    #w2 = x>d
    sum = sersic(x,a1,r1,n) + exponential(x,a2, r2)#sersic(x,a2,r2,1)#
    return sum

def log_err(intens_err,intens):
    y = abs(intens_err/(intens*np.log(10)))
    return y

def redshift_d(z):
    y = (3*10**5 * z)/73
    return y*1000

path = '/volumes/ssd/article_data/ngc12'
obj_name = 'NGC 12'
tbl = imp_tbl(path+'/test.csv')
z = 0.0131
#err, max_err, cut_idx, median_arr = iso_err_plot(path)
d = redshift_d(z)
z_p = 22.58
z_g = 22.54
cut_i = 0
cut = 1
sma0 = tbl['sma']
#pa = tbl['pa']
#eps = tbl['ellipticity']

intens_g = tbl['g']
intens_r = tbl['r']
#intens_r = tbl['intens']
#intens_err = tbl['intens_err']#/(1.89**2)
g_err = tbl['g_err']/(0.262**2)
r_err = tbl['r_err']/(0.262**2)

#print(log_err(intens_r[r_err==np.max(r_err)], np.max(r_err)));sys.exit()

#kpc = kpc(sma, d)
radius = kpc(sma0, d) #arcsec(sma) #
sma = radius

def init_param(x,intens,bt):
    
    i_0 = intens[1]
    i_hr = i_0 / np.exp(1)
    abs_i = abs(intens - i_hr)
    idx_d = list(abs_i).index(np.min(abs_i))
    r_s = sma0[idx_d]
    #print(r_s)
    
    d = bt*5*r_s#np.max(sma)*bt #
    bulge_sma_M = np.max(sma[sma<=d])
    bulge_half_flux = np.sum(intens[sma<=bulge_sma_M])/2
    
    abs_b = abs(intens-bulge_half_flux)
    flux_b = list(abs_b).index(np.min(abs_b))
    amp_b = intens[flux_b]
    r_eff_1 = sma[flux_b]
    
    r_eff_2 = (np.max(sma)-(np.min(sma)+d))/2 + d
    abs_d = abs(x-r_eff_2)
    sma_d = list(abs_d).index(np.min(abs_d))
    amp_d = intens[sma_d]
    rand = np.random.randint(0,10)/10 #noise

    return amp_b+rand,r_eff_1+rand,amp_d+rand,r_eff_2+rand,4, r_s#,d+rand
"""
def init_param(sma, intens, bt):
    d = np.max(sma)*bt
    bulge_sma_M = np.max(sma[sma<=d])
    bulge_half_flux = np.sum(intens[sma<=bulge_sma_M])/2
    
    abs_b = abs(intens-bulge_half_flux)
    flux_b = list(abs_b).index(np.min(abs_b))
    amp_b = intens[flux_b]
    r_eff_1 = sma[flux_b]
    
    disk_m = np.min(sma[sma>=d])
    sma_disk = sma[sma>=d]
    disk_half_flux = np.sum(intens[sma>=disk_m])/4
    amp_d = intens[0]
    
    abs_d = abs(disk_half_flux - intens)
    idx_d = list(abs_d).index(np.min(abs_d))
    #print(intens[idx_d])
    r_s = sma[idx_d]
    #print(r_s);sys.exit()
    rand = np.random.randint(0,10)/10 #noise
    return amp_b+rand,r_eff_1+rand,amp_d+rand,r_s+rand,4#,d+rand
"""
bt = 0.1

popt_raw = []
for i in range(2000):
    init_list= [init_param(sma,intens_r,bt)][0]
    #print(init_list);sys.exit() #amp_b,amp_d,r_eff_1,r_eff_2,n,r_s 
    popt, pcov = curve_fit(sum_profile, sma[cut_i:-cut],intens_r[cut_i:-cut],p0=init_list,maxfev=8000, sigma=log_err(r_err[cut_i:-cut],intens_r[cut_i:-cut]))
    popt_raw.append(popt)
popt_arr = np.array(popt_raw)
mena, popt,std = sigma_clipped_stats(popt_arr, axis=0, cenfunc='median',stdfunc='mad_std',sigma=3)#np.median(popt_arr,axis=0)
#print(popt)
a1,r1 = popt[0], popt[1]
a2, r2,n4,r_s = popt[2], popt[3],popt[4], popt[5]

#print(5*r_s)

fig, ax = plt.subplots(2,1,figsize=(5,7), gridspec_kw={'height_ratios':[5,2]}, sharex=True)
plt.subplots_adjust(hspace=0)

def tick(i):
    ax[i].xaxis.set_minor_locator(AutoMinorLocator(5))
    ax[i].yaxis.set_minor_locator(AutoMinorLocator(5))
    ax[i].tick_params(axis='y', which='minor', direction='out')
    ax[i].tick_params(axis='y', which='major', direction='out')
    ax[i].tick_params(axis='x', which='minor', direction='out')
    ax[i].tick_params(axis='x', which='major', direction='out')

ax[0].plot(sma, mag(sersic(sma,a2,r2,1),z_p), label='disk', linestyle='dashed',c='C0') # a2*np.exp(-(kpc/r2)),z_p
ax[0].plot(sma, mag(sersic(sma,a1,r1,n4),z_p), label='bulge', linestyle='dashed',c='C3') # a1*np.exp(-(kpc/r1)**(1/n)),z_p
#ax[0].plot(radius, mag(sersic(radius,a2,r2,n1),z_p)+mag(sersic(sma,a1,r1,4),z_p))
ax[0].plot(sma, mag(sum_profile(sma, *popt),z_p), label='sum', linestyle='dashdot',linewidth=1,c='C2')
#ax[0].fill_between(radius[:cut_idx], mag(max_err,z_p)-log_err(intens_err[:cut_idx], intens[:cut_idx]), mag(max_err,z_p)+err+log_err(intens_err[:cut_idx], intens[:cut_idx]), color='lightgrey') #2.5*np.log10(intens_err[:cut_idx])/2
#ax[0].scatter(radius[:cut_idx], mag(median_arr,z_p), s=2, color='orange')
ax[0].scatter(sma[cut_i:-cut], mag(intens_r[cut_i:-cut], z_p),s=5,c='orange')
#ax[0].errorbar(sma, mag(intens_r, z_p), yerr=log_err(r_err, intens_r), c='orange')
#err = log_err(r_err, intens_r)
#print(np.min(err[err>0]), np.max(err))
ax[0].set_title(obj_name) #title
ax[0].set_ylabel('$\mu_r$')
y_bot, y_top = np.min(mag(intens_r,z_p))-1,np.max(mag(intens_r, z_p))+1
ax[0].set_ylim(y_bot,y_top)
ax[0].text(1,y_top-(y_top-y_bot)*0.1, 'bulge(n='+f'{n4:.1f}'+') $R_{eff}=$'+f"{r1:.1f}kpc"+'\ndisk(n='+f'{1:.1f}'+') $R_{eff}=$'+f"{r2:.1f}kpc", bbox={'boxstyle':'square', 'fc':'white'})
ax[0].legend()
ax[0].invert_yaxis()
tick(0)

#ax[1].scatter(, mag(intens_g,z_g)-mag(intens_r,z_p))#, '.-')
ax[1].errorbar(sma,mag(intens_g,22.56)-mag(intens_r,22.59), yerr=log_err(r_err,intens_r)+log_err(g_err,intens_g))
ax[1].set_ylabel('g - r')
ax[0].axvline(x=r1, linestyle='dotted', linewidth=1.5, c='grey')
ax[1].axvline(x=r1, linestyle='dotted',linewidth=1.5, c='grey')
tick(1)

"""
ax[2].plot(kpc(sma,d), eps, '.-')
ax[2].set_ylabel('Ellipticity')
tick(2)
"""
fig.supxlabel('sma(kpc)')
plt.show()