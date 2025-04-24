#!/usr/bin/env python
# coding: utf-8

from sklearn.metrics import classification_report,auc,roc_auc_score
from PIL import Image
import time
from pathlib import Path

import os
import numpy as np
import pandas as pd
from PIL import Image

import warnings
warnings.filterwarnings('ignore', '.*do not.*', )
warnings.warn('Do not show this message')

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from glob import glob
#from skimage import io
from sklearn.utils import shuffle

#from nipype.interfaces.ants import N4BiasFieldCorrection
import sys

import warnings
warnings.filterwarnings('ignore')


def print_info(ims):

    for im in ims:
        
        print(im.shape, im.min(), im.max(), im.mean(), im.std(), \
              'n!=0:',im[im!=0].shape[0],im[im!=0].mean(),im.dtype)

def print_info_full(ims):

    for im in ims:
        
        print(im.shape, im.min(), im.max(), im.mean(), im.std(), \
              'n!=0:',im[im!=0].shape[0],'mean of!=0:',im[im!=0].mean(),im.dtype,
              'n uni:',len(np.unique(im.flatten())), 'quan 0.9:',
              np.quantile(im.flatten(), 0.9))



def show_n_images(imgs, cmap='gray', titles = None, enlarge = 4, mtitle=None,
                  cut = 0, axis_off = True, fontsize=15, cb = 0):

    plt.set_cmap(cmap);

    n = len(imgs);
    gs1 = gridspec.GridSpec(1, n);

    fig1 = plt.figure(figsize=(4*len(imgs),8));
    for i in range(n):

        ax1 = fig1.add_subplot(gs1[i]);
        if (cb):
            if len(np.unique(imgs[i])<=5):
                 img = imgs[i]
            else:

                img = cont_br(imgs[i])
        else:
            img = imgs[i]
        if cut:
            ax1.imshow(img[50:290, 75:450] , interpolation='none', origin='lower');
        else:

            ax1.imshow(img, interpolation='none');
        if (titles is not None):
            ax1.set_title(titles[i], fontsize=fontsize);  #, fontweight="bold");
        if (axis_off):
            plt.axis('off')
    if mtitle:
        plt.title(mtitle)
    plt.tight_layout()
    plt.show();

#------------------------------------------------------------------------------
# # Datasets utils

datasets=['spy2','spy1','duke']

# file extension per dataset
nifti_acq_ext_all={'spy2':'_spy2_vis1_dce_aqc_',
              'spy1':'_spy1_vis1_acq',
              'duke':'_duke_aqc_'}
mask_acq_ext_all={'spy2':'_spy2_vis1_mask',
                 'spy1':'_spy1_vis1_mask',
                 'duke':'_duke_mask'}

def get_dataset_from_id(pid):
    if 'ISPY1' in pid:
        return 'spy1'
    if 'ISPY2' in pid or 'ACRIN-6698' in pid:
        return 'spy2'
    if 'Breast_MRI' in pid:
        return 'duke'

base_path = None 
nifti_path = None
mask_path = None

def setup_paths(base, nifti_p, mask_p):
    
    
    global base_path, nifti_path, mask_path
    base_path = base
    nifti_path = nifti_p
    mask_path = mask_p
    
    print(nifti_path, mask_path)


# # Nifti utils

import nibabel as nib

def read_niftii(fname):
    # Load the NIfTI file
    nii_img = nib.load(fname)

    # Get the data (as a NumPy array)
    mnii_data = nii_img.get_fdata()

    return mnii_data

def get_nifti_acquisitions(pid ):

    global nifti_path
    
    dataset = get_dataset_from_id(pid)
    fpath=nifti_path[dataset]
    nifti_acq_ext=nifti_acq_ext_all[dataset]

    fname = pid + nifti_acq_ext + '0.nii.gz'
    #print(os.path.join(fpath,fname))
    if not os.path.isfile(os.path.join(fpath,fname)):
        print('no nifti files',os.path.join(fpath,fname) )
        return None

    x=read_niftii( os.path.join(fpath,fname))
    img=[x]
    for k in range(1,1000,1):

        fname=pid+nifti_acq_ext+str(k)+'.nii.gz'

        if not os.path.isfile(os.path.join(fpath,fname)):
            print('last acquisition', k)
            break

        x=read_niftii( os.path.join(fpath,fname))

        img.append(x)
    return img

def get_ser_acquisitions(pid, ser=[0,1,2]):

    global nifti_path
    dataset = get_dataset_from_id(pid)
    fpath=nifti_path[dataset]
    nifti_acq_ext=nifti_acq_ext_all[dataset]

    fname = pid + nifti_acq_ext +str(int(ser[0])) + '.nii.gz'

    if not os.path.isfile(os.path.join(fpath,fname)):
        print('no nifti files',os.path.join(fpath,fname) )
        return None

    x=read_niftii( os.path.join(fpath,fname))
    img=[x]
    fname = pid + nifti_acq_ext +str(int(ser[1])) + '.nii.gz'
    x=read_niftii( os.path.join(fpath,fname))
    img.append(x)
    fname = pid + nifti_acq_ext +str(int(ser[2])) + '.nii.gz'
    x=read_niftii( os.path.join(fpath,fname))
    img.append(x)

    return img

def get_nifti_acquisition(pid,idx=0):

    global nifti_path
    dataset = get_dataset_from_id(pid)
    fpath=nifti_path[dataset]
    nifti_acq_ext=nifti_acq_ext_all[dataset]

    fname = pid + nifti_acq_ext +str(int(idx)) + '.nii.gz'

    if not os.path.isfile(os.path.join(fpath,fname)):
        print('no nifti files',os.path.join(fpath,fname) )
        return None

    img=read_niftii( os.path.join(fpath,fname))
    

    return img

def get_nifti_mask(pid):

    dataset = get_dataset_from_id(pid)
    fpath=mask_path[dataset]
    mask_acq_ext=mask_acq_ext_all[dataset]

    fname = pid + mask_acq_ext + '.nii.gz'
    if not os.path.isfile(os.path.join(fpath,fname)):
        print('no nifti files')
        return None

    img=read_niftii( os.path.join(fpath,fname))

    return img


# # Mask utils

def find_first_last_planes(data):
    # Check for presence of '1' across the entire plane, reduce dimension with any()
    plane_has_one = np.any(data, axis=(1, 2))  # Reduce across the second and third dimensions

    # Find the indices of planes that contain at least one '1'
    indices = np.where(plane_has_one)[0]

    # Extract the first and last plane indices
    if indices.size > 0:
        first_plane = indices[0]
        last_plane = indices[-1]
        return int(first_plane), int(last_plane)
    else:
        return None, None  # or appropriate handling if no '1' is present in any plane
    
def get_nonzero_bounding_box(array_3d):
    """
    Find the minimum bounding box containing all non-zero elements in a 3D array.
    
    Parameters:
    array_3d (numpy.ndarray): 3D numpy array
    
    Returns:
    tuple: ((min_x, max_x), (min_y, max_y), (min_z, max_z)) - coordinates of the bounding box
           where min values are inclusive and max values are exclusive (Python-style slicing)
    """
    # Find non-zero element positions
    non_zero_positions = np.nonzero(array_3d)
    
    # If there are no non-zero elements, return None
    if len(non_zero_positions[0]) == 0:
        return None
    
    # Find min and max indices for each dimension
    min_x, max_x = np.min(non_zero_positions[0]), np.max(non_zero_positions[0]) + 1
    min_y, max_y = np.min(non_zero_positions[1]), np.max(non_zero_positions[1]) + 1
    min_z, max_z = np.min(non_zero_positions[2]), np.max(non_zero_positions[2]) + 1
    
    return ((min_x, max_x), (min_y, max_y), (min_z, max_z))


def minmax(im):
    if im.sum()==0:
        return im
    im = (im-im.min())/(im.max()-im.min())
    return im


# In[476]:


def to_rgb(a,b,c):
    x = np.stack([minmax(a),
                minmax(b),
                minmax(c)], axis=2)
    return x


def show_pid(pid):
    
    ds = get_dataset_from_id(pid)
    d=get_ser_acquisitions(pid, [0,1,2])
    print(pid,ds)
    
    m = get_nifti_mask(pid)
    if m is not None:
        s,e=find_first_last_planes(m)
        print(s,e,m.shape)
        show_n_images([to_rgb(d[0][k],d[1][k],d[2][k]) for k in np.linspace(s+1, e-1, num=5, dtype=int)],
                 titles=[pid+' plane '+str(int(k)) for k in np.linspace(s+1, e-1, num=5, dtype=int)])
        show_n_images([to_rgb(d[0][k],d[1][k],m[k]) for k in np.linspace(s+1, e-1, num=5, dtype=int)],
                     titles=[pid+' plane '+str(int(k)) for k in np.linspace(s+1, e-1, num=5, dtype=int)])
    else:
        print('==== No mask')
        show_n_images([to_rgb(d[0][k],d[1][k],d[2][k]) for k in np.linspace(5, d[0].shape[0]-5, num=5, dtype=int)],
                 titles=[pid+' plane '+str(int(k)) for k in np.linspace(5, d[0].shape[0]-5, num=5, dtype=int)])
        



