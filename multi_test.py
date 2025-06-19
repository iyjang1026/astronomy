import numpy as np
import astropy.io.fits as fits
from astropy.stats import sigma_clip
from multiprocessing import Process, JoinableQueue
import multiprocessing
import glob
import warnings
from scipy.stats import mode 
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
    obj, seg_map = sep.extract(subd, 2*bkg_rms, segmentation_map=True)
    mask_map = np.array(seg_map)
    kernel = skimage.morphology.disk(3) #np.array([[1,1,1],[1,1,1],[1,1,1]])
    mask_map_d = binary_dilation(mask_map, kernel, iterations=2)
    masked = np.where((mask_map_d!=0), np.nan, data0)
    return masked


warnings.filterwarnings('ignore')
def mask(path):
    file = glob.glob(path + '/N*.fits')
    data_list = []
    bar0 = progressbar.ProgressBar(maxval=len(file), widgets=['[',progressbar.Timer(),']',progressbar.Bar()]).start()
    for n in range(len(file)):
        data = se_mask(file[n])
        clip_data = sigma_clip(data, sigma_lower=6, sigma_upper=3, cenfunc='median', stdfunc='mad_std')
        mode1 = mode(data[~np.isnan(clip_data)])[0]
        mode_arr = (clip_data - mode1)/mode1
        data_list.append(mode_arr)
        bar0.update(n)
    bar0.finish()
    return np.array(data_list)



#bar = progressbar.ProgressBar(maxval=x,widgets=['[',progressbar.Timer(),']', progressbar.Bar()]).start()
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
    data_tot = mask('/media/iyjang/SSD/2025-05-12/r')
    #stop_event = multiprocessing.Event()
    tasks=[]
    q = JoinableQueue()
    for k in range(multiprocessing.cpu_count()):
        thread = Process(target=index, args=(data_tot, q))
        tasks.append(thread)
        thread.start()
    
    #stop_event.set()
    #print(f'stop signal')
    result = q.get()
    for task in tasks:
        #q.join()
        task.kill()
        task.join()
    data = np.array(result)
    fits.writeto('/media/iyjang/SSD/2025-05-12/test_dark_sky.fits', data, overwrite=True)
    print(f'processes finished')
    

if __name__ == '__main__':
    main() 