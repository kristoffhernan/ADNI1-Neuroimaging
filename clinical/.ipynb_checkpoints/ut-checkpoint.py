import ants
import glob
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from multiprocessing import Pool
import gc

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




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





def imgs_to_matrix(directory, wm_files=None, gm_files=None, combine=False):     
    imgs = []
    
    if combine:
        
        for file_grouping in zip(wm_files, gm_files):
            wm_path, gm_path = file_grouping
            
            wm_img = ants.image_read(os.path.join(directory, wm_path))
            gm_img = ants.image_read(os.path.join(directory, gm_path))

            # Clone the white matter image
            comb_img = ants.image_clone(wm_img)

            # Add the gray matter image to the combined image
            comb_img = comb_img + gm_img

            # grab subject and img info
            sub_id_wm, img_id_wm = wm_path.split('-')[1:3]
            sub_id_gm, img_id_gm = gm_path.split('-')[1:3]
            
            if sub_id_wm != sub_id_gm:
                raise ValueError(f'wm id:{sub_id_wm} ne to gm id:{sub_id_gm}')
                
            vector =  comb_img.numpy().ravel()
            
            # Calculate the volume of the brain
            voxel_size = 1.5  # in mm
            brain_vol = np.count_nonzero(vector) * (voxel_size ** 3)

            # turn to matrix, then to 1D array
            img_vec = np.append(vector, brain_vol)
            img_vec = np.append([sub_id_wm, img_id_wm], img_vec)
            imgs.append(img_vec)
        
        X = np.vstack(imgs)
        return X
        
    else: 
        all_files = [wm_files, gm_files]
        both_Xs = []
        for files in all_files:
            imgs = []
            for path in files:
                img = ants.image_read(os.path.join(directory, path))

                # grab subject and img info
                sub_id, img_id = path.split('-')[1:3]

                vector = img.numpy().ravel()
                
                # Calculate the volume of the brain
                voxel_size = 1.5  # in mm
                brain_vol = np.count_nonzero(vector) * (voxel_size ** 3)

                # turn to matrix, then to 1D array
                img_vec = np.append(vector, brain_vol)
                img_vec = np.append([sub_id, img_id], img_vec)
                imgs.append(img_vec)

            # stack the vectors into a 2D array
            X = np.vstack(imgs)
            both_Xs.append(X)
            
        # wm_X, gm_X
        return both_Xs
    
    

def matrix_to_df(X):
    # V: Voxel intensity
    # turn matrix to dataframe and name columns
    X_cols = ['Subject', 'Img_ID'] + ['V{}'.format(i+1) for i in range(X.shape[1]-2)]
    X_df = pd.DataFrame(X, columns=X_cols)
    
    return X_df

def clean_data(X_df, md):
    # merge two dfs
    X_cl = md.merge(X_df, on=['Subject', 'Img_ID'])
    X_cl = X_cl.drop(columns=['Img_ID', 'Subject'])

    # create X and y
    y = X_cl['Group'].values
    X = X_cl.drop(columns=['Group'])
    
    return X, y


def get_metadata(data_path):
    # Clean metadata dataframe
    md = pd.read_csv(os.path.join(data_path, 'ADNI1_Complete_2Yr_3T_4_18_2023.csv'))

    md = md.rename(columns={'Image Data ID': 'Img_ID'})

    md = md.drop(columns=['Visit', 'Modality', 'Description', 'Type', 'Acq Date', 'Format', 'Downloaded'])

    md['Group'] = md['Group'].map({'CN':0, 'MCI':1, 'AD':2})
    md['Sex'] = md['Sex'].map({'F':0, 'M':1})

    return md





def perform_pca(X, y):
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    
    # Preprocess and scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fit and transform the PCA model on the training set
    pca = PCA(random_state=10)

    X_train_pca = pca.fit_transform(X_train_scaled)

    # calculate cumulative variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # find the index where cumulative variance reaches 95%
    n_components = np.argmax(cumulative_variance >= 0.95) + 1
    print(f'n components:{n_components}')
    
    # re-fit PCA with the chosen number of components
    pca = PCA(n_components=n_components, random_state=10)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Transform the test set using the trained PCA model
    X_test_pca = pca.transform(X_test_scaled)
    
    return X_train_pca, y_train, X_test_pca, y_test


def perform_logreg(X_train_pca, y_train, X_test_pca, y_test):
    clf = LogisticRegression(random_state=10, penalty=None, multi_class='multinomial', max_iter=1000).fit(X_train_pca, y_train)
    y_preds = clf.predict(X_test_pca)
    
    acc = sum(y_preds == y_test) / len(y_test)
    
    return acc


def full_pipeline(X_matrix):
    
    md = get_metadata(data_path)

    df = matrix_to_df(X_matrix)
    
    X, y = df.pipe(clean_data, md=md)
    
    X_train_pca, y_train, X_test_pca, y_test = perform_pca(X, y)
    
    return X_train_pca, y_train, X_test_pca, y_test