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


import os

def show_n_images_nsave(imgs, cmap='gray', titles=None, enlarge=4, mtitle=None,
                  cut=0, axis_off=True, fontsize=15, cb=0, imsave=0):
    plt.set_cmap(cmap)
    n = len(imgs)
    gs1 = gridspec.GridSpec(1, n)
    fig1 = plt.figure(figsize=(4*len(imgs), 8))
    
    for i in range(n):
        ax1 = fig1.add_subplot(gs1[i])
        if (cb):
            if len(np.unique(imgs[i])) <= 5:
                img = imgs[i]
            else:
                img = cont_br(imgs[i])
        else:
            img = imgs[i]
        if cut:
            ax1.imshow(img[50:290, 75:450], interpolation='none', origin='lower')
        else:
            ax1.imshow(img, interpolation='none')
        if (titles is not None):
            ax1.set_title(titles[i], fontsize=fontsize)
        if (axis_off):
            plt.axis('off')
    
    if mtitle:
        plt.suptitle(mtitle)
    
    plt.tight_layout()
    
    # Save images if requested
    if imsave:
        # Create directory if it doesn't exist
        save_dir = 'im_for_article'
        os.makedirs(save_dir, exist_ok=True)
        
        if titles is not None:
            # Save individual images
            for i in range(n):
                # Clean filename (remove special characters)
                filename = titles[i].replace(' ', '_').replace('/', '_').replace('\\', '_')
                filename = ''.join(c for c in filename if c.isalnum() or c in ['_', '-', '.'])
                filepath = os.path.join(save_dir, f"{filename}.png")
                
                # Create individual figure for each image
                fig_single = plt.figure(figsize=(6, 6))
                if (cb):
                    if len(np.unique(imgs[i])) <= 5:
                        img = imgs[i]
                    else:
                        img = cont_br(imgs[i])
                else:
                    img = imgs[i]
                
                if cut:
                    plt.imshow(img[50:290, 75:450], interpolation='none', origin='lower', cmap=cmap)
                else:
                    plt.imshow(img, interpolation='none', cmap=cmap)
                
                plt.title(titles[i], fontsize=fontsize)
                if axis_off:
                    plt.axis('off')
                
                plt.tight_layout()
                #plt.savefig(filepath, dpi=300, bbox_inches='tight')
                #plt.close(fig_single)
                #print(f"Saved: {filepath}")
        
        # Save the combined figure
        if mtitle:
            combined_filename = mtitle.replace(' ', '_').replace('/', '_').replace('\\', '_')
            combined_filename = ''.join(c for c in combined_filename if c.isalnum() or c in ['_', '-', '.'])
        else:
            combined_filename = 'combined_figure'+str(imsave)
        
        combined_filepath = os.path.join(save_dir, f"{combined_filename}_combined.png")
        fig1.savefig(combined_filepath, dpi=300, bbox_inches='tight')
        print(f"Saved combined figure: {combined_filepath}")
    
    plt.show()
    
def show_n_images(imgs, cmap='gray', titles = None, enlarge = 4, mtitle=None,
                  cut = 0, axis_off = True, fontsize=15, cb = 0,imsave=0):

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
            ax1.set_title(titles[i], fontsize=fontsize); 
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
    
def save_niftii(fname, np_img):
 
    converted_array = np.array(np_img, dtype=np.float64)
    affine = np.eye(4)
    nifti_file = nib.Nifti1Image(converted_array, affine)
    
    nib.save(nifti_file, fname)
   
def get_nifti_acquisitions(pid ):

    global nifti_path
    
    dataset = get_dataset_from_id(pid)
    fpath=nifti_path[dataset]
    nifti_acq_ext=nifti_acq_ext_all[dataset]

    fname = pid + nifti_acq_ext + '0.nii.gz'
    print('loading',os.path.join(fpath,fname))
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
        print('no nifti files',pid,fpath,fname,nifti_acq_ext ,os.path.join(fpath,fname) )
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
        print('no nifti files', os.path.isfile(os.path.join(fpath,fname)))
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
        return None, None 
        
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

def to_rgb_m(a,b,c):
    x = minmax(np.stack([a,
                b,
                c], axis=2))
    return x


def show_pid(pid, ser=[0,1,2], bbox=None):
    
    ds = get_dataset_from_id(pid)
    
    d=get_ser_acquisitions(pid, ser)
    print(pid,ds, 'num axquisitions:',len(d))
    
    m = get_nifti_mask(pid)
    if m is not None:
        print('mask:',m.shape, sum(m[m>0].astype(np.int8)))
        s,e=find_first_last_planes(m)
        
        show_n_images([to_rgb(d[0][k],d[1][k],d[2][k]) for k in np.linspace(s+1, e-1, num=5, dtype=int)],
                 titles=[pid+' plane '+str(int(k)) for k in np.linspace(s+1, e-1, num=5, dtype=int)])
        show_n_images([to_rgb(d[0][k],d[1][k],m[k]) for k in np.linspace(s+1, e-1, num=5, dtype=int)],
                     titles=[pid+' plane '+str(int(k)) for k in np.linspace(s+1, e-1, num=5, dtype=int)])
        show_n_images([m[k] for k in np.linspace(s+1, e-1, num=5, dtype=int)],
                     titles=[pid+' plane '+str(int(k)) for k in np.linspace(s+1, e-1, num=5, dtype=int)])
    if bbox is not None:
        print('==== No mask showing BoundingBox')
        m=np.zeros(d[0].shape)
        #m[int(bbox[0]):int(bbox[1]), int(bbox[2]):int(bbox[3]),  int(bbox[4]):int(bbox[5])]=1
        m[int(bbox[0]):int(bbox[1]), int(bbox[2]):int(bbox[3]),  int(bbox[4]):int(bbox[5])]=1
        idx=np.linspace(int(bbox[0])+1, int(bbox[1])-1, num=5, dtype=int)

        show_n_images([np.stack([minmax(d[0][k]),
                                minmax(d[1][k]),
                                m[k]],axis=2) for k in idx],axis_off=False)
        show_n_images([d[1][k] for k in idx])
        
    
        
def show_box(pid, df, show_bw=0, show_rgb=0):
    
    idx = df[df.pid==pid].index.values[0]
    
    r=df.loc[idx]
    pid=r['pid']
    print(pid)
    
    a0=get_nifti_acquisition(pid, idx=0)
    a1=get_nifti_acquisition(pid, idx=1)
    a2=get_nifti_acquisition(pid, idx=r['post_late']) # last acqisition
    startm=int(r['mask_start'])
    endm=int(r['mask_end'])

    sraw=int(r['sraw'])
    eraw=int(r['eraw'])

    scol=int(r['scol'])
    ecol=int(r['ecol'])
    print(startm,endm,sraw,eraw,scol,ecol,a0.shape)
    m = np.zeros(a0.shape)
    m[startm:endm,sraw:eraw,scol:ecol]=1
    idx=[startm+1, (startm+endm)//2,endm-1]
    if show_rgb:
        show_n_images([ds.to_rgb(a0[k],a1[k],a2[k]) for k in idx], axis_off=False)
        show_n_images([np.stack([ds.minmax(a0[k]) ,
                                ds.minmax(a1[k]) ,
                                m[k]],axis=2 ) for k in idx])
    if show_bw:
        show_n_images([a0[k] for k in idx], axis_off=False)
        show_n_images([a1[k] for k in idx], axis_off=False)
        show_n_images([a2[k] for k in idx], axis_off=False)
####  ML
from sklearn.metrics import roc_curve, auc
def plot_roc(y, y_pred, tlt=''):

    #scores = clf.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y, y_pred)

    plt.figure(figsize=(3,3))
    plt.plot(fpr, tpr)

    plt.xlabel('FPR (False Positive Rate = 1-specificity)')
    plt.ylabel('TPR (True Positive Rate = sensitivity)')
    print(tlt , ' AUC ROC score ' ,str(np.round(roc_auc_score(y, y_pred),3)))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(tlt+' ROC = '+str(np.round(roc_auc_score(y, y_pred),3)));

from sklearn.metrics import precision_recall_fscore_support,accuracy_score,confusion_matrix
from sklearn.metrics import classification_report,auc,roc_auc_score
#import vis_utils as vs


def report_full(y, y_pred, tlt = 'Train',classes=['Negative', 'pCR pos'],
                c_rep = 0, Thresh=0.5, plot_roc_c=0, plot_cm=0):

    t = '*******   '+tlt+' *******'
    #display(HTML('<font size=3>'+t+'</font>'))
    #print(t)
    y_int_pred = y_pred.copy()

    if (len(y_pred.shape)>1):

        y_pred = y_pred[:,1].copy()

    y_int_pred = np.where(y_pred>Thresh, 1,0)
    acc = accuracy_score(y, y_int_pred)
    auc=np.around(roc_auc_score(y, y_pred),3)

    labs = [i for i in range(len(classes))]

    cm = confusion_matrix(y, y_int_pred,  labels=labs)

    if (len(classes) ==2):

        tn, fp, fn, tp = cm.ravel()

        specificity = tn / (tn+fp)
        sensitivity = tp/(tp+fn)
        precision = tp / (tp + fp)  # Positive Predictive Value (PPV)
        npv = tn / (tn + fn)  # Negative Predictive Value (NPV)

        print('\n',tlt, ' Accuracy: ',np.around(acc,3),
              ' AUC: ',np.around(auc,3) ,
              '  Specificity:', np.around(specificity,3),
              '  Sensitivity:',
              np.around(sensitivity,3),
              'NPV:',np.around(npv,3),
              'Precision:', np.around(precision, 3))



    if c_rep:

        print(classification_report(y, y_int_pred,  labels=labs, target_names=classes))

    #print(title)

    #fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
    pd.DataFrame(cm)
    print(pd.DataFrame(cm, columns=['No','Yes']).head())
    if plot_cm:


        #axes[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plot_cm(cm, classes)

    if plot_roc_c:
        plot_roc(y, y_pred, tlt=tlt)
        '''fpr, tpr, thresholds = roc_curve(y, y_pred)


        axes[1].plot(fpr, tpr)
        axes[1].title(tlt+' ROC = '+str(np.around(auc,3)));'''


