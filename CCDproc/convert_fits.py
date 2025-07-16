import numpy as np
import astropy.io.fits as fits
import os

class convert_fits:
    def __init__(self, path):
        import glob
        self.path = glob.glob(path + '/*.fits') #import all '.fits' files in path
        self.file = glob.glob(path) #import single '.fits' file. it must have file name and path
    
    def mkdir(path, folder):
        if os.path.exists(path + folder):
            pass
        else:
            os.mkdir(path + folder)
    def debayer_RGGB_single(path): #debayer the single image
        import numpy as np
        from astropy.io import fits

        data = fits.open(path)[0]
        hdu = data.data
        hdr = data.header
        hdu.astype(np.float32)
        import colour_demosaicing
        t = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(hdu, pattern='RGGB') #change pattern same with yours
        raw, column, color = t.shape

        r = []
        g = []
        b = []

        for i in range(0,raw):
            column_r = t[i].T[0]
            r.append(column_r)
            column_g = t[i].T[1]
            g.append(column_g)
            column_b = t[i].T[2]
            b.append(column_b)

        img = np.array([r, g, b], dtype=np.float32)
        return img, hdr
        
        #print(img.shape)

    def debayer_RGGB_multi(path): #debayer multiple images in directory of path
        files = convert_fits(path + '/pp_obj').path
        convert_fits.mkdir(path,'/color')
            
        for i in files:
            img, hdr= convert_fits.debayer_RGGB_single(i)
            root, file = os.path.split(i)
            fits.writeto(path +'/color/debayered_'+file, img, hdr, overwrite=True)


    def split_rgb_single(path, obj_name): #split rgb single image
        
        data = fits.open(path + obj_name + '.fits')[0]
        hdu = data.data
        hdr = data.header
        print(hdu.shape)
        
        r, g, b = hdu #split tri-colour
        
        hdr.append(('color', 'r', ''))
        fits.writeto(path+obj_name+'r.fits', r.astype(np.float32), header=hdr, overwrite=True)
            
        hdr.update({'COLOR':'g'})
        fits.writeto(path+obj_name+'g.fits', g.astype(np.float32), header=hdr, overwrite=True)
            
        hdr.update({'COLOR':'b'})
        fits.writeto(path+obj_name+'b.fits', b.astype(np.float32), header=hdr, overwrite=True)
        

    def split_rgb_multi(path, obj_name): #splite rgb mulitple images
        
        color = ['r', 'g', 'b']
        for i in color:
            folder = '/' + i
            convert_fits.mkdir(path, folder)

        r_path = path + '/r/'
        g_path = path + '/g/'
        b_path = path + '/b/'

        path = convert_fits(path + '/color').path #

        for i in range(len(path)):
            data = fits.open(path[i])[0]
            hdu = data.data
            hdr = data.header
            n = format(i, '04')
            r, g, b = hdu #split tri-colour

            hdr.append(('color', 'r', ''))
            fits.writeto(r_path+obj_name+str(n)+'_r.fits', r.astype(np.float32), header=hdr, overwrite=True)
            
            hdr.update({'COLOR':'g'})
            fits.writeto(g_path+obj_name+str(n)+'_g.fits', g.astype(np.float32), header=hdr, overwrite=True)
            
            hdr.update({'COLOR':'b'})
            fits.writeto(b_path+obj_name+str(n)+'b.fits', b.astype(np.float32), header=hdr, overwrite=True)

#convert_fits.debayer_RGGB_multi('/volumes/ssd/2025-06-29')
#convert_fits.split_rgb_multi('/volumes/ssd/2025-06-29/color_flat', 'flat')
