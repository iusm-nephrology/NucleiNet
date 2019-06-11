# Config parameters

general 
- name - the name of the experiment  
- n_gpu - number of gpus (0 for cpu or 1 GPU), program will decide if GPU is available or not

arch
- type - the name of the model in models.py

data_loader
- type - name of data loader in data_loaders.py
- args
   - data_dir - directory with images
   - csv_path - path to .csv file with all the image names and labels in the training set
   - batch_size - number of images per batch, used to improve training speed and decrease overfitting
   - shuffle - whether to shuffle the order of images
   - validation_split - fraction of images to use during training to evaluate the model performance
   - num_workers - improves speed by preparing batches prior to input into the model
   - training - true if training
   
data_loader_test
- type - name of same data loader as data_loader
- args
   - data_dir - directory with images
   - csv_path - path to .csv file with all the image names and labels in the test set
   - batch_size - number of images per batch, used to improve training speed and decrease overfitting
   - shuffle - whether to shuffle the order of images
   - validation_split - needs to be 0.0 for testing
   - num_workers - improves speed by preparing batches prior to input into the model
   - training - false for testing
   
optimizer
- type - name of optimizer, see the use of [torch.optim](https://pytorch.org/docs/stable/optim.html)
- args
   - lr - learning rate (how much to adjust the netowrk based on the round of back propogation)
   - weight_decay - how much the learning rate should decrease per epoch
   - amsgrad - whether or not to use the amsgrad version of the ADAM optimizer

loss - the loss function to use to calculate back propogation in loss.py

metrics - string list of metrics defined in metrics.py

lr_scheduler 
- type - the name of the learning rate schedule (see torch.optim)
- args
   - step_size - number of epochs between learning rate adjustments
   - gamma - multiplicative factor of learning rate decay
   
trainer
- epochs - number of training iterations
- save_dir - where the logs from the training are saved
- save_period - number of epochs between saves
- verbosity - amount of detail printed to the screen. Use 0 for faster speeds
- monitor - a string with the function and the parameter separated by a space. deafult is minimum value of the loss function on validation data
- early_stop - number of epochs with no improvement in the monitored value at which training is terminated
- tensorboardX - visualization service, not implemented currently
- log_dir - where the log should save training information
