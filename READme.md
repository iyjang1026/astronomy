Astronomy
==========

python package dependency
---
pip install <package name>

numpy, glob, os, astropy, photutils, ccdproc

# create env astro

for mac
---
```bash
conda create -n astro python=3.12 --platform osx-64 #formac
```

for linux
---
```bash
conda create -n astro python=3.12 --platform linux-64 #for linux
```

conda forge setting
---
```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
```

# astromatic

astromatic's softwares is very useful. these software need shell baed command like bash(linux), zsh(mac)

how to install?
---
```bash
conda install conda-forge astromatic::<astromatic-software>
```
source-extractor, swarp, scamp, psfex

# astrometry.net

initial astrometry tool

how to install?
---
```bash
conda install conda-forge::astrometry
```

code info
---
this is astronomical data processing code based on python<3.12

this code has pre-processing with bias, dark, flat(either dark-sky flat) and sky subtraction.

 
