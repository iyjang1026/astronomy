import numpy as np
from astropy.table import Table
from astropy.modeling.models import Sersic1D
from astropy.stats import sigma_clipped_stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from isolist_plot import iso_err_plot
def imp_tbl(path):
    tbl = Table.read(path, format='ascii.csv')
    return tbl

def sersic(x,amp, r_eff, n):
    return Sersic1D(amplitude=amp, r_eff=r_eff, n=n)(x)

def exponential(x, amp, r_eff):
    return amp*np.exp(-(x/r_eff))

def mag(x, z_p):
    return -2.5*np.log10(x)+z_p

def arcsec(x):
    return (x*1.89)#
def kpc(x,d):
    return d * np.tan((np.pi/180)*((x*1.89)/3600))

def sum_profile(x,a1,r1,n4,a2,r2,n1):
    sum = sersic(x,a1,r1,n4) + sersic(x, a2, r2,1) #a1*np.exp(-((x/r1)**(1/n)))+a2*np.exp(-(x/r2))#
    return sum

def log_err(intens_err,intens):
    y = abs(intens_err/(intens*np.log(10)))
    return y

path = '/volumes/ssd/intern/25_summer/M51_L'
tbl = imp_tbl(path+'/tbl_median_test_1/tbl_-0.0.csv')
err, max_err, cut_idx, median_arr = iso_err_plot(path)
d = 8580
z_p = 28.44
cut = 38
sma = tbl['sma']
pa = tbl['pa']
eps = tbl['ellipticity']
#print(sma)
intens = tbl['intens']/(1.89**2)
intens_err = tbl['intens_err']#/(1.89**2)
#kpc = kpc(sma, d)
radius = kpc(sma, d) #arcsec(sma) #
popt, pcov = curve_fit(sum_profile, radius[cut:cut_idx], median_arr[cut:cut_idx],sigma=err[cut:cut_idx]+log_err(intens_err[cut:cut_idx], intens[cut:cut_idx]), maxfev=8000)
print(popt)
a1,r1,n4 = popt[0], popt[1], popt[2]
a2, r2,n1 = popt[3], popt[4], popt[5]
fig, ax = plt.subplots(3,1,figsize=(5,7), gridspec_kw={'height_ratios':[5,1,1]}, sharex=True)
plt.subplots_adjust(hspace=0.07)

def tick(i):
    ax[i].xaxis.set_minor_locator(AutoMinorLocator(5))
    ax[i].yaxis.set_minor_locator(AutoMinorLocator(5))
    ax[i].tick_params(axis='y', which='minor', direction='out')
    ax[i].tick_params(axis='y', which='major', direction='out')
    ax[i].tick_params(axis='x', which='minor', direction='out')
    ax[i].tick_params(axis='x', which='major', direction='out')

ax[0].plot(radius, mag(sersic(radius,a2,r2,n1),z_p), label='disk', linestyle='dashed',c='C0') # a2*np.exp(-(kpc/r2)),z_p
ax[0].plot(radius, mag(sersic(radius,a1,r1,n4),z_p), label='bulge', linestyle='dashed',c='C3') # a1*np.exp(-(kpc/r1)**(1/n)),z_p
#ax[0].plot(radius, mag(sum_profile(radius, *popt),z_p), label='sum', linestyle='dotted')
ax[0].fill_between(radius[:cut_idx], mag(max_err,z_p)-log_err(intens_err[:cut_idx], intens[:cut_idx]), mag(max_err,z_p)+err+log_err(intens_err[:cut_idx], intens[:cut_idx]), color='lightgrey') #2.5*np.log10(intens_err[:cut_idx])/2
ax[0].scatter(radius[:cut_idx], mag(median_arr,z_p), s=2, color='orange')
ax[0].set_title('NGC 5195') #title
ax[0].set_ylabel('$\mu_r$')
ax[0].set_ylim(18,30)
ax[0].text(1,28.5, 'bulge(n='+f'{n4:.1f}'+') $R_{eff}=$'+f"{r1:.1f}kpc"+'\ndisk(n='+f'{n1:.1f}'+') $R_{eff}=$'+f"{r2:.1f}kpc", bbox={'boxstyle':'square', 'fc':'white'})
ax[0].legend()
ax[0].invert_yaxis()
tick(0)

ax[1].plot(kpc(sma,d), pa, '.-')
ax[1].set_ylabel('PA(deg)')
tick(1)

ax[2].plot(kpc(sma,d), eps, '.-')
ax[2].set_ylabel('Ellipticity')
tick(2)
fig.supxlabel('sma(kpc)')
plt.show()