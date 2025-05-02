import glob
import os
def astrometry(path, obj_name, ra, dec, radius):
    file = open(path+'/'+obj_name+'.sh', 'w')
    file.write(f'solve-field --index-dir /Users/jang-in-yeong/solve/index4100 --use-source-extractor -3 {ra} -4 {dec} -5 {radius} --no-plots')
    file.close()

path = str(input('path:'))
if path == None:
    path = os.getcwd()
obj_name = str(input('obj_name:'))
ra = str(input('ra:'))
dec = str(input('dec:'))
rad = str(input('radius(deg):'))

astrometry(path, obj_name, ra, dec, rad)