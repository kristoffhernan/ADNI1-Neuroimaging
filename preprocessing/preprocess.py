import psutil
import gc
import os
import pandas as pd
import zipfile
import numpy as np
import nibabel as nib
from io import BytesIO
import matplotlib.pyplot as plt
import nilearn as nil # https://nilearn.github.io/stable/quickstart.html
import time
from nilearn import plotting
from tqdm import tqdm
from functools import partial
import multiprocessing as mp

# from multiprocessing import Pool
# import torch
# from torch.utils import data
import ants
from antspynet import brain_extraction
from scipy.ndimage import rotate

import io

import multiprocessing
from joblib import Parallel, delayed
import concurrent.futures as cf



# learn to parallelize later

class Pipeline():
    def __init__(self, path):
        self.path = path
        self.zip_len = self.__zip_len__()
        self.files = self.get_files()

        
        
    def __zip_len__(self):
        '''
        Returns:
            zip_len: length of the zipfile
        '''
        
        zp = zipfile.ZipFile(self.path, 'r')
        zip_len = len(zp.namelist())
        
        return(zip_len)
    
    
    def get_files(self):
        '''
        Returns:
            files: file paths of all files inside of zip
        '''
        
        with zipfile.ZipFile(self.path, 'r') as zp:
            files = zp.namelist()
        zp.close()

        return(files)
    
    
    def get_img_info(self, start=None, stop=None):
        '''
        Parameters:
            start: file index to start from
            stop: file idex to end at

        Returns:
            ls: a list of tuples of the subject id and img id of an MRI scan
        '''

        ls = []
        
        if start and stop is None:
            start = 0
            stop = self.zip_len
            
        for file in self.files[start:stop]:
            # file contains many folders, split and grab the folder pertaining to subject and img id
            Subject_ID = file.split('/')[1]
            Img_ID = file.split('/')[5].split('_')[-1].split('.')[0]
            ls.append((Subject_ID, Img_ID))
        
        return ls
    
    
    def get_images(self, start=0, stop=None):
        '''
        Parameters:
            start: file index to start from
            stop: file idex to end at

        Returns:
            imgs (array): array of a lists where each list contains the ANTs object and its information in a tuple
        '''
        if stop is None:
            stop = self.zip_len
        
        imgs = []
        
        # open up zipfile in archive, but don't extract everything
        with zipfile.ZipFile(self.path, 'r') as zp: 
            # loop through files and open to read them in  
            for idx, file in enumerate(self.files[start:stop]):
                
                # check that the file exists
                try:
                    binary_file = zp.open(file, 'r').read()
                    bb = BytesIO(binary_file)
                    fh = nib.FileHolder(fileobj=bb)

                    # read in using nibabel
                    x = nib.Nifti1Image.from_file_map({'header': fh, 'image': fh})

                    # convert to ants object to preprocess
                    x = ants.utils.from_nibabel(x)
                    
                except KeyError:
                    print(f"File {file} does not exist in the archive.")
                
                Subject_ID, Img_ID = self.get_img_info(start, stop)[idx]

                imgs.append([x, (Subject_ID, Img_ID)])
                
                
                # update on memeory data takes 
                gc.collect()
                process = psutil.Process()
                memory_info = process.memory_info()
                
                if (stop - start <= 15):
                    print(f'Image {idx + 1} loaded')
                if (idx % 20 == 0):
                    print(f"After loading {idx + 1} images, Memory used: {memory_info.rss / 1024 / 1024:.2f} MB")

                    
        zp.close()
        print(f'Read in {stop - start} images')
        
        return np.array(imgs, dtype=object)
    
        

        
    def n4(self, x):
        '''
        Parameters:
            x: some ANTS image 
            
        Returns:
            x: intensity corected image
        '''
        # image histogram
        hist, bins = np.histogram(x.numpy().ravel(), bins=256)

        # mean intensity value of the brain tissue
        mean_intensity = bins[125:175].mean()

        # Set the lower and upper thresholds to exclude the empty space
        low_threshold = mean_intensity + 0.1 * mean_intensity
        high_threshold = np.amax(x.numpy()) - 0.1 * np.amax(x.numpy())

        # binary mask of the brain region
        mask = ants.get_mask(x, low_thresh=low_threshold, high_thresh=high_threshold, cleanup=2)

        # n4 bias
        x = ants.n4_bias_field_correction(x, 
                                        shrink_factor=3, 
                                        mask=mask, 
                                        convergence={'iters':[20, 10, 10, 5],
                                                     'tol':1e-07},
                                        rescale_intensities=True)
        
        return x
    
    
    
    def normalize(self, x):
        '''
        Parameters:
            x: some ANTS image 
            
        Returns:
            x: intensity normalized image by z standard normal
        '''
        x = x.iMath_normalize()
        
        return x
    
    
    def resize(self, x, dim=tuple):
        '''
        Parameters:
            x: some ANTS image 
            dim: tuple of 3 values to resample image to
            
        Returns:
            x: resampled image
        '''
        # apply the transformation to the image
        x = ants.reflect_image(x, axis=1)
        
        # change spacing/voxel size
        # (1.5,1.5,1.5) is the voxel size of the template
        # (0.9) to (1.2) for x,y,z is ADNI voxel/spacing size
        target_spacing = (1.5,1.5,1.5)
        x.set_spacing(target_spacing)
        
        # unique shapes of ADNI images
        # {(256, 256, 170), 
        # (256, 256, 160), 
        # (256, 256, 166), 
        # (248, 256, 160), 
        # (240, 256, 160)}
    
        # resample image to be smallest size
        smallest_dim = (240, 256, 160)
        x = x.resample_image(smallest_dim, use_voxels=True, interp_type=1)
        
        # crop image to remove black
        # cropping changes the dimension
        x = ants.crop_image(x)
        
        # change dim to template dim
        # interp_type= 1 (nearest neighbor)
        # could find another template with more similar dimesions
        x = x.resample_image(dim, use_voxels=True, interp_type=1)
        x.set_spacing(target_spacing)
        
        target_origin = (0.0, 0.0, 180.0)
        x.set_origin(target_origin)

        return x
    
    
    
    def skull_strip_img(self, x, weight='t1'):
        '''
        Parameters:
            x: some ANTS image 
            weight: t1 or t2 depending on the weight of the MRI
            
        Returns:
            skstr_img_t1: skull stripped image
        '''
        # create a binary mask, with 1's being brain and 0's ow
        brain_mask = brain_extraction(x, modality=weight)
        skstr_img_t1 = x * brain_mask
        
        return skstr_img_t1
    
    
    
    def segment_img(self, img):
        '''
        Parameters:
            img: some ANTS image 
            
        Returns:
            [gm, wm]: white matter and gray matter ANTS img
        '''
        # Segment brain into gray matter, white matter, and cerebrospinal fluid
        segs = ants.atropos(a=img, x=ants.get_mask(img), m='[0.1,1x1x1]', c='[2,0]', i='kmeans[1]')
        gm = segs['probabilityimages'][0]
        wm = segs['probabilityimages'][1]
        
        return [gm, wm]
    
 

    def smooth_img(self, x, sig=(8,8,8)):
        # Smooth the image with a Gaussian filter
        x = ants.smooth_image(x, sigma=sig, FWHM=True)
        return x
    

    
    def map_to_template(self, template_img, moving_img):
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

    
    
    def load_template(self, data_path):
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
    
   

    # def save_img(self, new_path, gm, wm, gm_sm, wm_sm, info):
    def save_img(self, new_path, gm_sm, wm_sm, gm_sm_m, wm_sm_m, gm_nsm_m, wm_nsm_m, info):

        '''
        Parameters:
            new_path: path to save image to, will create new directory if one isnt made
            gm: ANTS object of gray matter 
            wm: ANTS object of white matter
            info: subject and id info in a tuple or list to be added to file name
            
        Returns:
            registered_image: registered image, applied with rigid, affine and deformation transformations
        '''

        subject_id, img_id = info
        suffix = '.nii.gz'

        # creates directory if one isnt made
        directory = os.path.join(new_path, 'preprocessed', 'imgssss')
        if not os.path.exists(directory):
            os.makedirs(directory)


        # save gm and white matter to directory
        # not smoothed mapped to template
        ants.image_write(gm_nsm_m, os.path.join(directory,
                                          'nsmt-' + subject_id + '-' + img_id + '-gm' + suffix), ri=False)    
        ants.image_write(wm_nsm_m, os.path.join(directory,
                                          'nsmt-' + subject_id + '-' + img_id + '-wm' + suffix), ri=False)    

        # smoothed no mapped to template
        ants.image_write(gm_sm, os.path.join(directory,
                                          'snmt-' + subject_id + '-' + img_id + '-gm' + suffix), ri=False)    
        ants.image_write(wm_sm, os.path.join(directory,
                                          'snmt-' + subject_id + '-' + img_id + '-wm' + suffix), ri=False)
        
        # smoothed, mapped to template
        ants.image_write(gm_sm_m, os.path.join(directory,
                                          'smt-' + subject_id + '-' + img_id + '-gm' + suffix), ri=False)    
        ants.image_write(wm_sm_m, os.path.join(directory,
                                          'smt-' + subject_id + '-' + img_id + '-wm' + suffix), ri=False)

    
        print(f'saved preprocessed img: {img_id} for subject: {subject_id}')
        
        
        
    def full_preprocessing(self, data_path, start=0, stop=None, return_ls=True): 
        '''
        Parameters:
            data_path: path where template is located
            start: index to start preprocessing
            stop: index to stop preprocessing
            return_ls: True/False to return the preprocessed images in a list 
        
        Returns:
        '''
        GM_tmpt, WM_tmpt = self.load_template(data_path)
        
        ls = []        
        
        if stop is None:
            stop = self.zip_len
        
        i = 0
        # all preprocessing steps, pretty self explanatory
        for x, info in tqdm(self.get_images(start, stop)):
            
            x = self.resize(x, dim=(121, 145, 121))
            
            x = self.n4(x)

            x = self.normalize(x)
                
            x = self.skull_strip_img(x, weight='t1')
    
            gm, wm = self.segment_img(x)
            
            # smoothed no mapped
            gm_sm = self.smooth_img(gm, sig=(8,8,8))
            wm_sm = self.smooth_img(wm, sig=(8,8,8))
            
            # smoothed mapped
            gm_sm_m = self.map_to_template(GM_tmpt, gm_sm)
            wm_sm_m = self.map_to_template(WM_tmpt, wm_sm)
            
            # not smoothed mapped
            gm_nsm_m = self.map_to_template(GM_tmpt, gm)
            wm_nsm_m = self.map_to_template(WM_tmpt, wm)
    
            self.save_img(data_path, gm_sm, wm_sm, gm_sm_m, wm_sm_m, gm_nsm_m, wm_nsm_m, info)

            
            gc.collect()
            process = psutil.Process()
            memory_info = process.memory_info()
            i += 1
            if i % 20 == 0:
              print(f"After preprocessing and saving {i + 1} images, Memory used: {memory_info.rss / 1024 / 1024:.2f} MB")
            
            # if true, append img to list
            if return_ls:
                ls.append(x)
        
        if return_ls:
            return ls
        else:
            return 'All images saved'
    
      
    

if __name__== '__main__':
    t1 = time.time()
    
    data_path = '/scratch/users/neuroimage/conda/data'
    full_path = os.path.join(data_path, 'ADNI1_Complete_2Yr_3T.zip')
    Pipe = Pipeline(full_path)
        
    results = Pipe.full_preprocessing(data_path, start=324, return_ls=False)

    t2 = time.time()
    
    print(f"The entire script took {(t2 - t1) / 60} minutes")
    
