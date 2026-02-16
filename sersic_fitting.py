import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.modeling.models import Sersic1D, custom_model
from astropy.modeling import FittableModel
from astropy.modeling.fitting import LMLSQFitter, TRFLSQFitter, DogBoxLSQFitter
from astropy.table import Table
import sys

def sersic(amp, r_eff,n):
    y = Sersic1D(amp, r_eff, n)
    return y
path = '~/data/ngc12'

tbl = Table.read(path+'/test.csv', format='ascii')
sma = tbl['sma']
r = tbl['r']
r_err = tbl['r_err']
g = tbl['g']
g_err = tbl['g_err']
cut = 7
#print(sersic(sma,36,5,4));sys.exit()

def init_param(x):
    d = np.max(x)*0.1
    r_eff_1 = (d-np.min(x))*(1/4)
    r_eff_2 = (np.max(x)-d)*(1/4)
    abs_b = abs(x-r_eff_1)
    sma_b = list(abs_b).index(np.min(abs_b))
    amp_b = r[sma_b]

    abs_d = abs(x-r_eff_2)
    sma_d = list(abs_d).index(np.min(abs_d))
    amp_d = r[sma_d]
    return amp_b,r_eff_1,amp_d,r_eff_2,d

print(init_param(sma))

@custom_model
def ser_exp(x,amp_b=1.,r_eff_1=1., amp_d=1.,r_eff_2=1.,d=1.):#, r_eff_3=1.):
    w1 = x<=d
    w2 = x>=d
    #print(sma_d);sys.exit()
    y = w1*Sersic1D(amp_b,r_eff_1,4)(x)+w2*amp_d*np.exp(-(x/r_eff_2))#+Sersic1D.evaluate(x,amp_d,r_eff_2,1) #
    return y

def log_err(intens_err,intens):
    y = abs(intens_err/(intens*np.log(10)))
    return y

amp_b,amp_d,r_eff_1,r_eff_2,d = init_param(sma)

sigma = tbl['r_err']
fitter = LMLSQFitter()
s_init = ser_exp(r_eff_1=r_eff_1,r_eff_2=r_eff_2,d=d)#amp_b=amp_b,amp_d=amp_d,
#print(s_init)

t = fitter(s_init, sma[:-cut], r[:-cut], maxiter=8000,weights=log_err(sigma[:-cut],r[:-cut]))
print(t)
plt.plot(sma, np.log10(t(sma)))
plt.plot(sma, np.log10(Sersic1D.evaluate(sma, amplitude=amp_b, r_eff=r_eff_1,n=4)), linestyle='dashed')
plt.plot(sma, np.log10(Sersic1D.evaluate(sma, amplitude=amp_d, r_eff=r_eff_2,n=1)), linestyle='dashed')
plt.plot(sma, np.log10(r), '.')
plt.show()
