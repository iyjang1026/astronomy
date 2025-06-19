from astropy.nddata import CCDData
import astropy.units as u
from astropy.stats import mad_std, sigma_clipped_stats
import astropy.io.fits as fits
from ccdproc import combine, subtract_bias, subtract_dark, flat_correct
import numpy as np
import glob
import os
class Ccdprc:
    def __init__(self, path):
        import glob
        self.path = glob.glob(path + '/*.fits') #import files in path
        self.file = glob.glob(path) #import file directly. it must have full name and path of file

    def combine_bias(path):
        file = glob.glob(path + '/BIAS/*.fits')
        biases = []
        for i in file:
            img_data = CCDData.read(i, unit=u.adu)
            biases.append(img_data)
        
        combined_bias = combine(img_list=biases,
                             method='median',
                             sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                             sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std, 
                            )
        combined_bias.meta['combined'] = True
        combined_bias.write(path + '/BIAS/combined_bias.fits')
        

    def combine_dark(path):
        file = glob.glob(path + '/DARK/*.fits')
        bias_img = CCDData(fits.open(path + '/BIAS/' + 'combined_bias.fits')[0].data, unit=u.adu)
        dark_image = []
        for i in file:
            img_data = CCDData.read(i, unit=u.adu)
            calibrated_dark = subtract_bias(img_data, bias_img)
            dark_image.append(calibrated_dark)

        combined_dark = combine(img_list=dark_image,
                                 method='median',
                                 sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                                 sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std,
                                )
        combined_dark.meta['combined'] = True
        combined_dark.write(path +'/DARK/combined_dark.fits')

    def DB_sub(path, obj_name):
        file = glob.glob(path + '/LIGHT/*.fits')
        bias_img = CCDData.read(path+'/BIAS/combined_bias.fits', unit=u.adu)
        dark_img = CCDData.read(path+'/DARK/combined_dark.fits', unit=u.adu)
        os.mkdir(path+'/LIGHT/DB_subed')
        for i in file:
            data = CCDData.read(i, unit=u.adu)
            sub_b = subtract_bias(data, bias_img)
            sub_d = subtract_dark(sub_b, dark_img, exposure_time='exposure', exposure_unit=u.second)
            n = format(file.index(i), '04')
            sub_d.write(path +'/LIGHT/DB_subed/p'+ obj_name + '_'+str(n)+'_'+'.fits')

    def masking_single(path):
        from photutils.background import Background2D, MedianBackground
        from astropy.convolution import convolve
        from photutils.segmentation import make_2dgaussian_kernel, detect_sources
        from astropy.stats import SigmaClip
        sub_d = fits.open(path)[0].data
        hdr = fits.open(path)[0].header
        sigma_clip = SigmaClip(sigma=3.0)
        bkg_estimator = MedianBackground(sigma_clip)
        bkg = Background2D(sub_d, (64,64), filter_size=(3,3), bkg_estimator=bkg_estimator)
        sub_d1 = sub_d - bkg.background

        threshold = 3*bkg.background_rms
        
        import skimage
        from scipy.ndimage import binary_dilation
        kernel = skimage.morphology.disk(3)
        #kernel = make_2dgaussian_kernel(3.0, size=3)
        convolved_data = convolve(sub_d1, kernel)

        segment_map = detect_sources(convolved_data, threshold, npixels=6)
        mask_map = np.array(segment_map)
        mask_map = binary_dilation(mask_map, kernel, iterations=2)
        masked = np.where((mask_map!=0), np.nan, sub_d)
        return masked, hdr
    
    def se_mask(file):
        import sep
        data = fits.open(file)[0].data
        hdr = fits.open(file)[0].header
        data1 = data.astype(data.dtype.newbyteorder('='))
        bkg_data = sep.Background(data1)
        bkg = bkg_data.back()
        bkg_rms = bkg_data.globalrms
        subd = data - bkg
        obj, seg_map = sep.extract(subd, 3*bkg_rms, segmentation_map=True)
        mask_map = np.array(seg_map)
        from scipy.ndimage import binary_dilation
        import skimage
        kernel = skimage.morphology.disk(3)
        mask_map_d = binary_dilation(mask_map, kernel, iterations=2)
        masked = np.where((mask_map_d!=0), np.nan, data)
        return masked, hdr
    
    def combine_dark_sky_flat(path1):
        import numpy as np
        import astropy.io.fits as fits
        from multiprocessing import Process, JoinableQueue
        import multiprocessing
        import glob
        import warnings
        from scipy.stats import mode
        import weakref
        import progressbar

        warnings.filterwarnings('ignore')

        def mask(path):
            file = glob.glob(path + '/N*.fits')
            scale_list = []
            mode0 = []
            bar0 = progressbar.ProgressBar(maxval=len(file), widgets=['[',progressbar.Timer(),']',progressbar.Bar()]).start()
            for n in range(len(file)):
                data = Ccdprc.se_mask(file[n])
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
    
        def main(path):
            multiprocessing.freeze_support()
            data_tot, mode_tot = mask(path)
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
            fits.writeto(path+'/dark_sky_flat.fits', data_final , overwrite=True)
            print(f'processes finished')
    

        if __name__ == '__main__':
            main(path1)

        
    
    def flat_correct(path, obj_name):
        file = glob.glob(path+'/LIGHT/DB_subed/r/N*.fits')
        flat = CCDData.read(path+'/LIGHT/DB_subed/r/fd.fits', unit=u.adu)
        os.mkdir(path+'/preprocessed_r1')
        from astropy.stats import sigma_clipped_stats
        for i in range(len(file)):
            data = CCDData.read(file[i], unit=u.adu)
            n = format(i, '04')
            mean, median, std = sigma_clipped_stats(data, cenfunc='median', stdfunc='mad_std')
            preprocessed = flat_correct(data, flat, norm_value=median)
            preprocessed.meta['preprocessed'] = True
            preprocessed.write(path + '/preprocessed_r1/pp'+obj_name+'_'+str(n)+'.fits')
        
            

#Ccdprc.combine_bias('/volumes/ssd/2025-05-25')
#Ccdprc.combine_dark('/volumes/ssd/2025-05-25')
#Ccdprc.DB_sub('/volumes/ssd/2025-05-25', 'NGC5907')
#Ccdprc.combine_dark_sky_flat('/volumes/ssd/2025-05-25')
#Ccdprc.mask('/volumes/ssd/2025-05-25')
#Ccdprc.flat_correct('/volumes/ssd/2025-05-25', 'NGC5907')