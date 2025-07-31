import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.modeling.models import Sersic1D
import sys
from scipy.optimize import curve_fit

tbl = Table.read('/volumes/ssd/intern/25_summer/M101_L/iso_integ_median_tbl.csv', format='ascii.csv')
sma = tbl['sma'] #장반경
intens = tbl['intens']/(1.89**2) #단위면적당 intens로 변환
pa = tbl['pa'] 
eps = tbl['ellipticity']


d = 6730
kpc = d * np.tan((np.pi/180)*((sma*1.89)/3600)) #장반경을 pixel 단위에서 kpc으로 변환

def sersic(x,amp,r_eff,n): #서직윤곽 함수
    de_v_r = Sersic1D(r_eff=r_eff,amplitude=amp, n=n)(x)
    return de_v_r

def mag(x): #등급으로 변환
    mag = -2.5*np.log10(x)+28.39
    return mag

param = [] #팽대부와 원반의 서직함수의 parameter를 담는 리스트
intens_l = [] #팽대부와 원반 각각을 fitting하기위해 나뉜 intens를 저장하는 리스트
sma_l = [] #intens_l 에 대응되는 장반경 값을 저장하는 리스트
def sum_profile(x,d): #팽대부와 원반의 서직함수를 더해서 전체 서직 함수를 얻는 함수, d는 팽대부와 원반을 구분하는 일종의 구분점
    inner_sma = kpc[kpc<=d] #팽대부 fitting용 장반경
    outer_sma = kpc[kpc>=d] #원반 fitting용 장반경
    inner_intens = intens[kpc<=d] #팽대부 fitting용 intens
    outer_intens = intens[kpc>=d] #원반 fitting용 intens
    
    intens_l.append(inner_intens)
    intens_l.append(outer_intens)
    sma_l.append(inner_sma)
    sma_l.append(outer_sma)
    popt_b, pcov_b = curve_fit(sersic, inner_sma, inner_intens, maxfev=7000) #팽대부 fitting
    popt_d, pcov_d = curve_fit(sersic, outer_sma, outer_intens, maxfev=7000) #원반 fitting
    param.append(popt_b)
    param.append(popt_d)
    sum = sersic(x, *popt_b)+sersic(x,*popt_d) #팽대부 + 원반
    return sum


popt_s, pcov_s = curve_fit(sum_profile, kpc, intens) #팽대부 + 원반 함수를 전체 intens에 맞게 fitting, 최적화 변수는 d
p_b, p_d = param[-2], param[-1] #팽대부와 원반 함수의 parameter
print(f'bulge amp, r_eff, n {p_b}')
print(f'disk amp, r_eff, n {p_d}')
print(f'sepert r, overray {popt_s}')

plt.plot(kpc, mag(sum_profile(kpc, *popt_s)), label='sum_profile') #sum profile
plt.plot(kpc, mag(sersic(kpc, *p_b)), label='bulge', linestyle='dashed')#bulge
plt.plot(kpc, mag(sersic(kpc, *p_d)), label='disk', linestyle='dashed')#disk
#plt.scatter(kpc, mag(intens), label='data', color='grey', s=3)
plt.scatter(sma_l[0], mag(intens_l[0]), color='C1', s=5)
plt.scatter(sma_l[1], mag(intens_l[1]), color='C2', s=5)
plt.xlabel('sma(kpc)')
plt.ylabel('$Mag_r$')
#plt.xscale('log', base=10)
#plt.yscale('log', base=10)
plt.ylim(28,18)
plt.legend()
plt.show()