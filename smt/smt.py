import ants
import glob
import os
import gc

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp


def get_ordered_files(directory, prefix):

    # Get a list of all files in the directory
    all_files = os.listdir(directory)

    # Filter files that start with prefix
    files = [f for f in all_files if f.startswith(prefix)]

    # Filter again for wm and gm
    gm_files = [f for f in files if f.endswith('-gm.nii.gz')]
    wm_files = [f for f in files if f.endswith('-wm.nii.gz')]

    # Sort the files list
    sorted_gm_files = sorted(gm_files)
    sorted_wm_files = sorted(wm_files)

    return sorted_gm_files, sorted_wm_files


def map_to_template(template_img, moving_img):
    '''
    Parameters:
        template_img: standard image to map to
        moving_img: the MRI images you want transformed

    Returns:
        registered_image: registered image, applied with rigid, affine and deformation transformations
    '''

    # Perform  registration
    synra_transform = ants.registration(fixed=template_img, moving=moving_img, 
                                         type_of_transform='SyNRA')

    # Apply the Rigid + Affine + deformable transform to the moving image
    # Forward transform: transforms points from the moving image to the fixed image 
    # Inverse transform: transforms points from the fixed image to the moving image
    registered_image = ants.apply_transforms(fixed=template_img, 
                                             moving=moving_img, 
                                             transformlist=synra_transform['fwdtransforms'])

    return registered_image

    
    
def load_template(data_path):
    # read in template, assign variable to wm and gm
    template_dir = os.path.join(data_path, '3DT1_AD_template_Priors')
    template_path = 'TPM_CnMciAD.nii'

    T1_std_template = ants.image_read(os.path.join(template_dir, template_path))

    GM_tmpt = ants.from_numpy(T1_std_template[:,:,:,0])
    WM_tmpt = ants.from_numpy(T1_std_template[:,:,:,1])

    # from_numpy changes the spacing to 1.0, so reset the spacing
    target_spacing = (1.5,1.5,1.5)
    GM_tmpt.set_spacing(target_spacing)
    WM_tmpt.set_spacing(target_spacing)

    # template images in in RAI orientation, change it to SAR like the ADNI images
    GM_tmpt = ants.reorient_image2(GM_tmpt, orientation='SAR')
    WM_tmpt = ants.reorient_image2(WM_tmpt, orientation='SAR')

    return [GM_tmpt, WM_tmpt]
    
   


data_path = '/scratch/users/neuroimage/conda/data'
img_path = os.path.join(data_path, 'preprocessed/imgsss')

snmt_files_gm, snmt_files_wm = get_ordered_files(img_path, "snmt")

GM_tmpt, WM_tmpt = load_template(data_path)
          
def process_file(i, file):
    img = ants.image_read(os.path.join(img_path, file))
    mapped = map_to_template(WM_tmpt, img)
    
    # smoothed, mapped to template
    ants.image_write(mapped, os.path.join(img_path,
                                           'smt' + file[4:]), 
                     ri=False)
    
    print(f"{i+1} file/s mapped")

    
num_processes = mp.cpu_count()
pool = mp.Pool(num_processes)

for i, file in enumerate(snmt_files_wm):
    pool.apply_async(process_file, args=(i, file))

pool.close()
pool.join()