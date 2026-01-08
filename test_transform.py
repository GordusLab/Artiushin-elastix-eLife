import glob, os, re, shutil, subprocess, pgzip, time, traceback, tifffile, numpy as np, numba as nb, \
    scipy.stats, joblib as jl
from itertools import chain
from skimage.measure import label, regionprops
import skimage.filters
from tqdm import tqdm




def findTransformix():
    try:
        v = subprocess.check_output(['transformix', '--version'])
        print('Found Transformix! {}'.format(v.decode('ascii').strip()))
        return 'transformix.exe'
    except FileNotFoundError as e:
        fnameTransformix = os.path.join(os.path.dirname(__file__), 'elastix_bin\\transformix.exe')
        if not os.path.exists(fnameTransformix):
            raise Exception('Could not find the Transformix executable! Aborting...')
        v = subprocess.check_output([fnameTransformix, '--version'])
        print('Found Transformix! {}'.format(v.decode('ascii').strip()))
        return fnameTransformix



fnameTransformix = findTransformix()


#path to folder with all relevant files
filefolder = 'Z:/Greg/transformix-test'

fMov = os.path.join(filefolder, 'moving.tif')

#should already be prepared with the proper InitialTransformParameterFileName order edited in
fnameTransform = os.path.join(filefolder, 'TransformParameters.1.txt')

fnameOutput = os.path.join(os.path.dirname(fnameTransform), 'result.tif')


# Get parameters
fileIn = fMov
fileOut = fnameOutput
fileTransform = fnameTransform
dirResults = os.path.dirname(fnameTransform)


os.makedirs(dirResults, exist_ok=True)


# Invoke elastix

output = ''
popen = subprocess.Popen([fnameTransformix, '-in', fileIn, '-tp', fileTransform, '-out', dirResults],
                             stdout=subprocess.PIPE, universal_newlines=True)
for stdout_line in iter(popen.stdout.readline, ""):
    output += stdout_line
print(output)
