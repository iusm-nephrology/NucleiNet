import argparse
import torch
import math
import argparse
import os, os.path, shutil
import pathlib
import random
import pandas as pd
import numpy as np
import h5py
import random
print("Modules loaded")

#python C:\Users\awoloshu\Documents\IMPRS\datasets\toHDF5.py -d C:\Users\awoloshu\Documents\IMPRS\datasets\F44_062419\allDAPI_volume -f mydata_test.h5

def list_files(directory):
    """Returns all files in a given directory
    """
    return [f for f in pathlib.Path(directory).iterdir() if f.is_file() and f.name.endswith('.csv')]

def list_dirs(directory):
    """Returns all directories in a given directory
    """
    return [f for f in pathlib.Path(directory).iterdir() if f.is_dir()]


def main(args):
    if args.dir == None: args.dir = "./"
    csv_files = list_files(args.dir)
    filename = args.filename
    if not (filename.endswith(".h5")): filename = filename + ".h5"
    filename = os.path.join(args.dir, filename)
    store = pd.HDFStore(filename)
    count = 0
    means = []
    stdevs = []
    num_images_train = 0
    num_images_test = 0
    for csv in csv_files:
        csv_path = os.path.join(args.dir, csv)
        print("Now loading: " + csv_path)
        csv_data = pd.read_csv(csv, header = None, engine='python')
        num_imgs = len(csv_data.index)
        imgs = list(range(num_imgs))
        random.shuffle(imgs)
        
        test_ind = imgs[0:int(num_imgs*args.split)]
        train_ind = imgs[int(num_imgs*args.split+1.0) :]
        
        img_start = 1
        if (csv_data.iloc[:,1].max() > 256).any:
            img_start = 2
            ids = csv_data.iloc[test_ind,1].to_numpy()
            ids = ids.reshape(ids.shape[0], 1)
            store.append('test_ids', pd.DataFrame(ids))
            
        train_img = csv_data.iloc[train_ind,img_start:] #first column is label
        
        train_img = train_img.dropna(axis=1)
        train_label = csv_data.iloc[train_ind,0]
        test_img = csv_data.iloc[test_ind,img_start:] #first column is label
        test_img = test_img.dropna(axis=1)
        test_label = csv_data.iloc[test_ind,0]
        if (train_img.min() < 0).any(): #convert signed to unsigned bytes
            train_img = train_img + 0
            test_img = test_img+0
            mask = train_img < 0
            train_img[mask] = train_img + 256
            mask = test_img < 0
            test_img[mask] = test_img + 256
            
        print(train_img.shape)
        print(test_img.shape)
        store.append('train_data', train_img)
        store.append('train_labels', train_label)
        store.append('test_data', test_img)
        store.append('test_labels', test_label)
        means.append(np.mean(train_img.to_numpy()))
        stdevs.append(np.std(train_img.to_numpy()))
        num_images_train = num_images_train + train_img.shape[0]
        num_images_test = num_images_test + test_img.shape[0]
        count = count +1
        del csv_data
        '''
        count = 0
        if count == 0:
            
            # Note: will generate warnings if filename contains a "."
            train_img.to_hdf(filename, 'train_data',mode='a', format='table')
            test_img.to_hdf(filename, 'test_data',mode='a', format='table')
            train_label.to_hdf(filename, 'train_labels',mode='a', format='table')
            test_label.to_hdf(filename, 'test_labels',mode='a', format='table')
            
            count = count
        else:
           
            train_img.to_hdf(filename, 'train_data',mode='a', append=True)
            test_img.to_hdf(filename, 'train_data',mode='a', append=True)
            train_label.to_hdf(filename, 'train_labels',mode='a', append=True)
            test_label.to_hdf(filename, 'test_labels',mode='a', append=True)
            '''
            

    metadata = pd.DataFrame({'TrainingNum': num_images_train, 'TestingNum': num_images_test, \
                'TrainingMean': np.mean(means), 'TrainingStd': np.mean(stdevs)}, index=[0])      
    store.append('Metadata', metadata)
    
    store.close()       
    with pd.HDFStore(filename) as f:
        print()
        print("Successfully created " + filename)
        print("===============")
        keys = list(f.keys())
        print("{0} keys in this file: {1}".format(len(keys), keys))
        print("Training images: " + str(num_images_train))
        print("Testing images: " + str(num_images_test))
        print("Training image mean = " + str(np.mean(means)) + " and std = " + str(np.mean(stdevs)))
   
        
            
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-d', '--dir', default=None, type=str,
                        help='csv folder (default: None)')
    parser.add_argument('-f', '--filename', default="dataset.h5", type=str,
                        help='name of file to save to (default: None)')
    parser.add_argument('-s', '--split', default=0.15, type=float,
                        help='split percentage for testing (default: None)')
    args = parser.parse_args()

    if args.dir:
        print("Converting .csv files to .h5 dataset...")
    else:
        raise AssertionError("Directory needs to be specified. Add '-d ./', for example.")


    main(args)
