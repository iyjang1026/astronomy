from astropy.io import fits
import sep
from photutils.segmentation import make_2dgaussian_kernel
from pysersic.priors import autoprior
from pysersic import FitSingle
from pysersic.loss import student_t_loss
import numpy as np
import sys

hdu = fits.open('~/data/ngc1064/NGC1064_r.fits')[0].data
hdu = hdu.astype(hdu.dtype.newbyteorder('='))
mask = fits.open('~/data/ngc1064/obj_rejec_NGC1064_r.fits')[0].data.astype(np.float32) 
mask = mask.astype(mask.dtype.newbyteorder('='))
bkg = sep.Background(hdu, mask=mask, bw=64, bh=64, fw=3, fh=3)
rms = bkg.rms()
prior = autoprior(image=hdu, profile_type='sersic_exp', mask=mask, sky_type='none')

psf = np.array(make_2dgaussian_kernel(3., size=3))
#sig = np.ones_like(hdu)
fitter = FitSingle(data=hdu, mask=mask,rms=rms, psf=psf, prior=prior, loss_func=student_t_loss)

from jax.random import (
    PRNGKey,
)
#print(PRNGKey(1000), PRNGKey(100))
map_params = fitter.find_MAP(rkey=PRNGKey(1000))
print(map_params)