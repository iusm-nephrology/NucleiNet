# MNIST_Example  
This example will show the organization and file structure for future deep learning projects. The Pytorch project template is based on the work done by [victoresque](https://github.com/victoresque/pytorch-template). 

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

## Updates
5/21/2019 Added examples for databases, dataloaders, configs, and Jupyter notebook for data in pickle format (CIFAR10), csv format (MNIST from CSV), and using image paths with csv metadata (traffic data). The traffic dataset will be most similar to actual implementation. 
- The traffic dataset is organized as a series of folders with images for training and testing. The csv file contains one column for the label and one column for the filename. The filename, image folder name, and label are all passed into the dataset class in databases.py. Training continues as expected. 
- Note: all Jupyter notebooks are NOT up to date. I only check their functionality through the training stage, not the testing stage. 
- the "data" and "saved" folders are not shown on this Github page due to memory concerns. New raw data should be saved as "data/datasetname/train/image001.jpg" and the csv file should be located at "data/datasetname/train.csv". The saved folder is described above and updated automatically. 
- At this point in time, mnist_example.ipynb contains the simplest example, while traffic_example.ipynb contains the most implementable example for our purposes
