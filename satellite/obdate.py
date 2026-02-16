import numpy as np
from astropy.table import Table
from astropy.io import fits
import glob
def date_csv(path):
    file = sorted(glob.glob(path+'/*.fits'))
    file_txt = open('/volumes/ssd/satelite/250604_UOU.txt', 'a')
    for i in range(len(file)):
        hdr = fits.open(file[i])[0].header
        file_txt.write(f'{hdr['CENOBSDATE']}\n')
    file_txt.close
date_csv('/volumes/ssd/satelite/stacked/250604_UOU_stacked')