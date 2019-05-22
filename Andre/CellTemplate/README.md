# Cell Template
This example will show the organization and file structure for future deep learning projects for nuclei classification. The Pytorch project template is based on the work done by [victoresque](https://github.com/victoresque/pytorch-template). 

Check the notebook for an example on training and testing.

## Explanation of structure
base -  contains generic models, data loaders, etc. that will be expanded upon in different examples.     
data_loader - used for creating a dataset, validation splitting, batch processing.   
model - define the model (LeNet architecture in this example), metrics (accuracy and top3 accuracy), and loss function (NLL Loss).   
trainer - a class for training and logging traning metrics.   
util - contains the logger base class as well as visualization using tensorboardX (optional).   
config - a file to show which data_loader, model, loss, optimization, trainer, gpu, etc. are being used.   
saved - a folder showing experiments, saved under the "name" config.json, then mmdd_HHMMSS timestamp automatically by train.py  
data - data to be used in the dataset   

## Customization
This structure can be changed by adding classes data_loaders.py to accomodate new datasets, new architectures in models.py, and hyperparameter optimization in config.json. 

Customizable options via config file
- number of GPUs
- model architecture
- data validation split and batch size
- data augmentation (translate, rotate, scale, crop, etc.)
- optimizer, including learning rate, weight decay, and type
- learning rate scheduler
- loss function
- number of epochs
- early stopping criteria

Coming soon
- data normalization
- class activation mapping
- support for 3D images
