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


class templateDataset(Dataset):
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
    
class groundTruthDataset(Dataset):
    def __init__(self, csv_path, root_dir, transforms=None):
        self.csv_data = pd.read_csv(csv_path, header = 0, engine='python')
        self.csv_data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
        self.root_dir = root_dir
        self.transforms = transforms
        self.data_len = len(self.csv_data)

        all_labels = self.csv_data.iloc[:,1]
        count_labels = {}
        for item in all_labels:
            if item: 
                key = "class_" + str(item)
                if not key in count_labels: count_labels[key] = 0
                count_labels[key] += 1 
        weights = []
        for key in count_labels:
            weights.append(1. / count_labels[key])
        self.weight = torch.FloatTensor(weights)
        
    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir,
                                str(self.csv_data.iloc[index, 0])) #path data is in the 1rst column
        image = io.imread(img_name)
        img_as_img = Image.fromarray(image) #convert to PIL
        label = int(self.csv_data.iloc[index, 1])  #label is in the 2nd column
        label = label - 1 # go from 1,2 to 0,1 --> not great solution
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img)
        return img_as_tensor, label
        
    
    def __len__(self):
        return self.data_len

    
