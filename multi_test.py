import numpy as np
import astropy.io.fits as fits
from astropy.stats import sigma_clip
from multiprocessing import Process, Queue, JoinableQueue
import multiprocessing
import glob
import warnings
from scipy.stats import mode
import matplotlib.pyplot as plt
import weakref
import sep
from scipy.ndimage import binary_dilation
import skimage
import progressbar

def se_mask(file):
    data0 = fits.open(file)[0].data
    data = np.array(data0, dtype=float)
    data1 = data.astype(data.dtype.newbyteorder('='))
    bkg_data = sep.Background(data1)
    bkg = bkg_data.back()
    bkg_rms = bkg_data.globalrms
    subd = data - bkg
    obj, seg_map = sep.extract(subd, 3*bkg_rms, segmentation_map=True)
    mask_map = np.array(seg_map)
    kernel = np.array([[1,1,1],[1,1,1],[1,1,1]]) #skimage.morphology.disk(3)
    mask_map_d = binary_dilation(mask_map, kernel, iterations=2)
    masked = np.where((mask_map_d!=0), np.nan, data0)
    return masked


warnings.filterwarnings('ignore')
def mask(path):
    file = glob.glob(path + '/N*.fits')
    scale_list = []
    mode0 = []
    bar0 = progressbar.ProgressBar(maxval=len(file), widgets=['[',progressbar.Timer(),']',progressbar.Bar()]).start()
    for n in range(len(file)):
        data = se_mask(file[n])
        mode1 = mode(data[~np.isnan(data)])[0]
        scaled_data = (data - mode1)/mode1
        scale_list.append(scaled_data)
        mode0.append(mode1)
        bar0.update(n)
    bar0.finish()
    mode_tot = np.median(np.array(mode0))
    return np.array(scale_list), mode_tot

def index(arr,q):
    l,x,y = arr.shape
    data_zero = np.zeros((x,y))
    for i in range(x):
        for j in range(y):
            arr_data = arr[:,i,j]
            weak_data = weakref.ref(arr_data)()
            data_zero[i,j] += np.median(weak_data[~np.isnan(weak_data)])
    q.put(weakref.ref(data_zero)())
    
def main():
    multiprocessing.freeze_support()
    data_tot, mode_tot = mask('/volumes/ssd/2025-06-16/LIGHT/DB_subed/r')
    tasks=[]
    q = JoinableQueue()
    for k in range(multiprocessing.cpu_count()):
        thread = Process(target=index, args=(data_tot, q))
        tasks.append(thread)
        thread.start()
    
    result = q.get()
    for task in tasks:
        task.kill()
        task.join()
    data_final = np.array(result)*mode_tot + mode_tot
    fits.writeto('/volumes/ssd/2025-06-16/test_dark_sky.fits', data_final , overwrite=True)
    print(f'processes finished')
    

if __name__ == '__main__':
    main()
