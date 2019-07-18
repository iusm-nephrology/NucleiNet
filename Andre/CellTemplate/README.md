# Cell Template
A tool for classifying nuclei in 2D or 3D images

This example will show the organization and file structure for future deep learning projects. The Pytorch project template is based on the work done by [victoresque](https://github.com/victoresque/pytorch-template). 

## upate 07/18/19
- alpha version 0.1 uploaded
- current datasets contain > 6000 images
- test accuracy > 90% 
- 2D and 3D images can be used

## update 06/06/2019
- added statistics and visualization packages in util
- implemented pretrained networks in model using resnet on upsampled images
- train using weight balanced loss function for imbalanced training data
- added a learning rate finder function to determine the learning rate with the best loss after one batch of training on the validation set

## update 05/30/2019
- trained on 517 images
- neural network options can be seen in groundTruth.config
- augmentation choices can be seen in dataloaders.py
- tested on 90 images with accuracy of 83.33%
- forward probability of training dataset for class 1 is 62.33% 
