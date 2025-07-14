import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.morphology import data_properties
import sep
from scipy.ndimage import binary_dilation

def se_mask(single_name):
        data = fits.open(single_name)[0].data
        data1 = data.astype(data.dtype.newbyteorder('='))
        bkg = sep.Background(data1)
        bkg_rms = bkg.rms()
        mean, median, std = sigma_clipped_stats(bkg_rms, cenfunc='median', stdfunc='mad_std', sigma=3)
        subd = data - bkg
        obj, seg_map = sep.extract(subd, 9.0*bkg.globalrms, segmentation_map=True) 
        mask_map = np.array(seg_map)
        kernel = np.array([[1,1,1],[1,1,1],[1,1,1]]) #skimage.morphology.disk(3)
        mask_map_d = binary_dilation(mask_map, kernel, iterations=2)
        masked = np.where((mask_map_d!=0), np.nan, data)
        return masked.astype(np.float32)


def data_proper(path):
    hdu = se_mask(path) #fits.open(path)[0].data #
    mean, median, std = sigma_clipped_stats(hdu, cenfunc='median', stdfunc='mad_std', sigma=3)
    data = hdu - median
    mask = data < 50
    cat = data_properties(data)
    columns = ['label','xcentroid','ycentroid','semimajor_sigma','semiminor_sigma','orientation']
    tbl = cat.to_table(columns=columns)
    print(tbl)
    import astropy.units as u
    from photutils.aperture import EllipticalAperture
    xypos = (cat.xcentroid, cat.ycentroid)
    print(xypos)
    r = 1  # approximate isophotal extent
    a = cat.semimajor_sigma.value * r
    b = cat.semiminor_sigma.value * r
    theta = cat.orientation.to(u.rad).value
    apertures = EllipticalAperture(xypos, a, b, theta=theta)
    plt.imshow(hdu, origin='lower', cmap='viridis',
           interpolation='nearest')
    apertures.plot(color='C3')
    plt.show()
data_proper('/volumes/ssd/intern/25_summer/M101_L/pp_obj/ppM1010000.fit')
