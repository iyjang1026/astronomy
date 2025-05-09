from astropy.nddata import CCDData, fits_ccddata_reader
import astropy.units as u
from astropy.stats import mad_std
import astropy.io.fits as fits
from ccdproc import combine, subtract_bias, subtract_dark, flat_correct, ccd_process
import numpy as np
import glob
import os
class Ccdprc:
    def __init__(self, path):
        import glob
        self.path = glob.glob(path + '/*.fits') #import files in path
        self.file = glob.glob(path) #import file directly. it must have full name and path of file

    def combine_bias(path):
        file = glob.glob(path + '/biases/*.fit')
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
        combined_bias.write(path + '/biases/combined_bias.fits')
        

    def combine_dark(path):
        file = glob.glob(path + '/darks/*.fit')
        bias_img = CCDData(fits.open(path + '/biases/' + 'combined_bias.fits')[0].data, unit=u.adu)
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
        combined_dark.write(path +'/darks/combined_dark.fits')

    def DB_sub(path, obj_name):
        file = glob.glob(path + '/lights/*'+obj_name+'*.fit')
        bias_img = CCDData.read(path+'/biases/combined_bias.fits', unit=u.adu)
        dark_img = CCDData.read(path+'/darks/combined_dark.fits', unit=u.adu)
        os.mkdir(path+'/lights/DB_subed')
        for i in file:
            data = CCDData.read(i, unit=u.adu)
            sub_b = subtract_bias(data, bias_img)
            sub_d = subtract_dark(sub_b, dark_img, exposure_time='exposure', exposure_unit=u.second)
            n = format(file.index(i), '04')
            sub_d.write(path +'/lights/DB_subed/p'+ obj_name + '_'+str(n)+'_'+'.fits')

    def masking_single(path):
        from photutils.background import Background2D, MedianBackground
        from photutils.segmentation import SourceFinder
        from astropy.convolution import convolve
        from photutils.segmentation import make_2dgaussian_kernel

        sub_d = fits.open(path)[0].data
        hdr = fits.open(path)[0].header

        bkg_estimator = MedianBackground()
        bkg = Background2D(sub_d, (64,64), filter_size=(3,3), bkg_estimator=bkg_estimator)
        sub_d1 = sub_d - bkg.background

        threshold = 3 * bkg.background_rms

        kernel = make_2dgaussian_kernel(1.6, size=33)
        convolved_data = convolve(sub_d1, kernel)

        finder = SourceFinder(npixels=10, progress_bar=False)
        segment_map = finder(convolved_data, threshold)

        mask_map = np.array(segment_map)
        smoothed = convolve(mask_map, kernel)
        masked = np.where((smoothed!=0), np.nan, sub_d)
        #final = np.where((masked==np.nan), np.median(np.isnan(masked)), masked)
        return masked, hdr
    
    def combine_dark_sky_flat(path):
        import glob
        file = glob.glob(path + '/lights/DB_subed/p*.fits')
        flat = []
        for i in range(len(file)):
            hdu,hdr = Ccdprc.masking_single(file[i])
            data = CCDData(hdu, unit=u.adu, header=hdr)
            flat.append(data)
            print(f'appended {i}')
        combined_flat = combine(img_list=flat, method='median', sigma_clip=True, 
                                sigma_clip_high_thresh=3, sigma_clip_low_thresh=6, 
                                sigma_clip_dev_func=mad_std, sigma_clip_func=np.ma.median)
        combined_flat.meta['combined'] = True
        combined_flat.meta['IMAGETYP'] = 'FLAT'
        combined_flat.write(path + '/lights/fd.fits')
        print(f'dark sky flat is made')
    
    def flat_correct(path, obj_name):
        file = glob.glob(path+'/lights/DB_subed/p*.fits')
        flat = CCDData.read(path+'/lights/fd.fits', unit=u.adu)
        os.mkdir(path+'/preprocessed')
        for i in range(len(file)):
            data = CCDData.read(file[i], unit=u.adu)
            n = format(i, '04')
            preprocessed = flat_correct(data, flat)
            preprocessed.meta['preprocessed'] = True
            preprocessed.write(path + '/preprocessed/pp'+obj_name+'_'+str(n)+'.fits')
    def mask(path):
        file = glob.glob(path + '/lights/DB_subed/*.fits')
        os.mkdir(path + '/masked_test')
        for i in range(len(file)):
            n = format(i, '04')
            hdu, hdr = Ccdprc.masking_single(file[i])
            data= CCDData(hdu, unit=u.adu, header=hdr)
            data.meta['masked'] = True
            data.write(path+'/masked_test/test'+str(n)+'.fits')


       
            

#Ccdprc.combine_bias('/volumes/ssd/intern/2024-12-01G')
#Ccdprc.combine_dark('/volumes/ssd/intern/2024-12-01G')
#Ccdprc.DB_sub('/volumes/ssd/intern/2024-12-01G', 'Abell2634')
#Ccdprc.combine_dark_sky_flat('/volumes/ssd/intern/2024-12-01G')
#Ccdprc.flat_correct('/volumes/ssd/intern/2024-12-01G', 'Abell2634')