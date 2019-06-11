from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader import databases
import os
import importlib


class augmentation_handler():
    """
    Take augmentation values from config and apply them to the data loader
    """
    def __init__(self, crop = None, rotation = 0, translate = 0.0, random_flip=0, scale= 0.0, resize = 28):
        #TODO fix resize options
       if crop:
           common_transforms = transforms.Compose([transforms.CenterCrop(crop), transforms.ToTensor()])
           trsfm_train = transforms.Compose([
              transforms.RandomHorizontalFlip(p = random_flip),
              transforms.RandomAffine(rotation, translate=(translate, translate), scale=(1-scale,1+scale), shear=None, resample=False, fillcolor=0),
              transforms.CenterCrop(crop),
              transforms.ToTensor()
              ])
       else:
           common_transforms = transforms.Compose([transforms.Resize(resize), transforms.ToTensor()])
           trsfm_train = transforms.Compose([
              transforms.Resize(resize),
              transforms.RandomHorizontalFlip(p = random_flip),
              transforms.RandomAffine(rotation, translate=(translate, translate), scale=(1-scale,1+scale), shear=None, resample=False, fillcolor=0),
              transforms.ToTensor()

            ])
       trsfm_test = common_transforms
       self.transforms = {'train': trsfm_train, 'test': trsfm_test}
                          
       super(augmentation_handler, self).__init__()

        
