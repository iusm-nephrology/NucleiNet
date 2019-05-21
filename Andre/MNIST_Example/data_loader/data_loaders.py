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

        
class Mnist_from_CSV_DataLoader(BaseDataLoader):
    def __init__(self, dataset1, batch_size, shuffle = True, validation_split = 0.0, num_workers =1, training = True):
        trsfm = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = dataset1
        #print(os.path.isfile('data/MNIST/mnist_train.csv'))
        importlib.reload(databases) #used to get load any recent changes from database class
        self.dataset = databases.MnistCSVDataset('data/MNIST/mnist_train.csv', transforms =trsfm)
        self.data_dir = databases.MnistCSVDataset(dataset1, transforms = trsfm)
        print("look something is here")
        super(Mnist_from_CSV_DataLoader, self).__init__(self.data_dir, batch_size, shuffle, validation_split, num_workers)
        
        
class cifar10_DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        self.data_dir = data_dir
        importlib.reload(databases) #used to get load any recent changes from database class
        self.dataset = databases.Cifar10Dataset('data/CIFAR10/cifar-10-batches-py/data_batch_1' ,transforms = trsfm)
        super(cifar10_DataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        
        
class traffic_DataLoader(BaseDataLoader):
    def __init__(self, csv_path, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
        ])
        self.data_dir = data_dir
        self.csv_path = csv_path
        importlib.reload(databases) #used to get load any recent changes from database class
        self.dataset = databases.trafficDataset(csv_path, data_dir, transforms=trsfm)
        super(traffic_DataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)