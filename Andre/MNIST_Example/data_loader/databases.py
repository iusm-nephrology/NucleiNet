from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image


class Cifar10Dataset(Dataset):
    def unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    def __init__(self, batch_path, transforms = None):
        self.data_info = self.unpickle(batch_path)
        X = self.data_info[b'data']
        Y = self.data_info[b'labels']
        self.img_arr = X.reshape(10000,3072)
        self.label_arr = np.array(Y)
        self.data_len = len(self.label_arr)
        self.transforms = transforms
        
    def __getitem__(self, index):
        img = self.img_arr[index, :]
        img_as_np = np.transpose(np.reshape(img,(3, 32,32)), (1,2,0))
        img_as_np = img_as_np.astype('uint8')
        single_image_label = self.label_arr[index]
        # Convert image from numpy array to PIL image, mode 'L' is for grayscale
        img_as_img = Image.fromarray(img_as_np)
         # Transform image to tensor
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img)
        # Return image and the label
        return img_as_tensor, single_image_label
                                                                 
    def __len__(self):
         return self.data_len                                                         
                                                               

class MnistCSVDataset(Dataset):
    def __init__(self, csv_path, transforms=None):
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 0])
        # Next 784 are the pixel values
        self.img_arr = np.asarray(self.data_info.iloc[:, 1:])
        # Calculate len
        self.data_len = len(self.data_info.index)
        self.transforms = transforms
        
    def __getitem__(self, index):
        single_image_label = self.label_arr[index]
        img_as_np = np.asarray(self.data_info.iloc[index][1:]).reshape(28, 28).astype('uint8')
        # Convert image from numpy array to PIL image, mode 'L' is for grayscale
        img_as_img = Image.fromarray(img_as_np)
        img_as_img = img_as_img.convert('L')
         # Transform image to tensor
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img)
        # Return image and the label
        return img_as_tensor, single_image_label
    
    def __len__(self):
        return self.data_len


class trafficDataset(Dataset):
    def __init__(self, csv_path, root_dir, transforms=None):
        self.csv_data = pd.read_csv(csv_path, header = 0)
        self.root_dir = root_dir
        self.transforms = transforms
        self.data_len = len(self.csv_data)
        
    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir,
                                self.csv_data.iloc[index, 7]) #path data is in the 7th column
        image = io.imread(img_name)
        img_as_img = Image.fromarray(image) #convert to PIL
        label = self.csv_data.iloc[index, 6]  #label is in the 6th column
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img)
        return img_as_tensor, label
        
    
    def __len__(self):
        return self.data_len

    