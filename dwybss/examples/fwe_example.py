'''
Created on 28 Sep 2017

@author: Miguel Molina Romero
@contact: miguel.molina@tum.de
@license: LPGL
'''
from dwybss.bss import bss_cals
import numpy as np
import os
import shutil

# Define paths
dpath = os.path.join(os.getcwd(), 'data')
opath = os.path.join(os.getcwd(), 'output')
shutil.rmtree(opath, True)
os.makedirs(opath)

# Define data and priors
obss = bss_cals.BssCals()
max_sources = 2    
tes = np.array([0.0751, 0.1351])
tes = tes[:, np.newaxis]
dwi = [os.path.join(dpath,'data_TE1.nii.gz'), os.path.join(dpath,'data_TE2.nii.gz')]
data = {'dwi': dwi, 'te': tes}
mask = os.path.join(dpath,'mask.nii.gz')
b_values = os.path.join(dpath,'test.bval')

# Run BSS
niiout = obss.fwe(data, mask, b_values, opath)

print(niiout)