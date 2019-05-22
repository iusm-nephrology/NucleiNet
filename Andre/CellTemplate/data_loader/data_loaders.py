from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader import databases
import os
import importlib


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super(MnistDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

        
        
        
class template_DataLoader(BaseDataLoader):
    def __init__(self, csv_path, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
        ])
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.batch_size = batch_size
        importlib.reload(databases) #used to get load any recent changes from database class
        self.dataset = databases.templateDataset(csv_path, data_dir, transforms=trsfm)
        super(template_DataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        
class groundTruth_DataLoader(BaseDataLoader):
    def __init__(self, csv_path, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm_train = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #random values, should be changed based dataset
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(45, translate=(.2,.2), scale=(0.75,1.25), shear=None, resample=False, fillcolor=0),
            #transforms.RandomRotation(45),
            transforms.CenterCrop(28),
            transforms.ToTensor(),
        ])
        trsfm_test = transforms.Compose([
            transforms.CenterCrop(28),
            transforms.ToTensor(),
        ])
        
        if training == True:
            trsfm = trsfm_train
        else:
            trsfm = trsfm_train       
        
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.batch_size = batch_size
        importlib.reload(databases) #used to get load any recent changes from database class
        self.dataset = databases.groundTruthDataset(csv_path, data_dir, transforms=trsfm)
        super(groundTruth_DataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)