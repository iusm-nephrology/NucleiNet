import os
import argparse
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from train import get_instance
import numpy as np
import torchvision
from torch.nn import functional as F
from torch import topk
import skimage.transform
from torch.optim import lr_scheduler
from tqdm import tqdm
import math
from utils import util
import pandas as pd
from sklearn.linear_model import LogisticRegression
from base import BaseDataLoader
from utils import transforms3d as t3d
import importlib
from data_loader import databases
from sklearn.preprocessing import scale, MinMaxScaler

# this code will take a .pth file, remove the last layer, train a logistic regression, then test 

class hdf5_3d_dataloader(BaseDataLoader):
    '''
    3D augmentations use a random state to peform augmentation, which is organized as a List rather than torch.Compose()
    '''
    def __init__(self, hdf5_path, batch_size, shuffle=True, shape = [7,32,32], validation_split=0.0, num_workers=1, training=True, mean = None, stdev = None):
        rs = np.random.RandomState()
        if mean is None or stdev is None:
            mean = 0
            std = 1
            print("No mean and std given!")
        
        trsfm = [t3d.Normalize(mean, stdev), t3d.ToTensor(True)]  
        
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.shape = shape
        importlib.reload(databases) #used to get load any recent changes from database class
        self.dataset = databases.hdf5dataset(hdf5_path, shape = self.shape, training = training, transforms=trsfm)
        super(hdf5_3d_dataloader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class SaveOutput:
    def __init__(self):
        self.outputs = []
        
    def __call__(self, module, module_in, module_out):
        to_save = module_out.view(module_out.size(0), -1).cpu().data.numpy()
        #to_save = module_out.cpu().data.numpy()
        self.outputs.append(to_save)
        return module_out
        
    def clear(self):
        self.outputs = []

def reassign_labels(list_of_things2change, class_old, class_new):
    list_return = []
    for mylist in list_of_things2change:
        for i, pred in enumerate(mylist):
            if isinstance(pred, np.ndarray): #topK
                for j, p, in enumerate(pred):
                    if p == class_old:
                        mylist[i,j] = class_new
            else:
                if pred == class_old:
                    mylist[i] = class_new
        list_return.append(mylist)
    return list_return

def main(config, resume):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("GPUs available: " + str(torch.cuda.device_count()))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    #data_loader = get_instance(module_data, 'data_loader', config) --> cant use this because it will auto split a validation set

    data_loader = hdf5_3d_dataloader(
        config['data_loader']['args']['hdf5_path'],
        batch_size=128,
        shuffle=False,
        validation_split=0.0,
        training=True,
        mean = config['data_loader']['args']['mean'], 
        stdev = config['data_loader']['args']['stdev']
    )

    print(len(data_loader.dataset))
    data_loader_test = get_instance(module_data, 'data_loader_test', config)

    # Load trained model, including fully connected
    model = get_instance(module_arch, 'arch', config)
    print(model)
    if torch.cuda.is_available():
        print("Using GPU: " + torch.cuda.get_device_name(0))
    else:
        print("Using CPU to test")
    loss_fn = getattr(module_loss, config['loss'])
    criterion = loss_fn(None)
    # load state dict
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    classes = ('S1', 'PCT', 'TAL', 'DCT', 'CD', 'CD45', 'Nestin', 'CD31_glom', 'CD31_inter')


    #identify feature layer to separate
    save_output = SaveOutput()
    #print(model.module.conv_layer4)
    handle = model.module.fc6.register_forward_hook(save_output)

    #generate feature set
    all_true = []
    feature_set = []
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            all_true.extend(target.cpu().data.numpy())
    print(i*128)
    all_true = np.array(all_true)
    all_features = np.array(save_output.outputs)

    for item in all_features:
        feature_set.extend(item)
    all_features = np.array(feature_set)
    #all_features = np.reshape(all_features, (feature_shape[0]*feature_shape[1], feature_shape[2]))

    print(all_true.shape)
    print(all_features.shape)
    #create + train logistic model
    minmaxscaler = MinMaxScaler()
    all_features_scaled = minmaxscaler.fit_transform(all_features)
    clf = LogisticRegression(random_state=0, class_weight='balanced', verbose = 1, solver = 'lbfgs', n_jobs=-1).fit(all_features_scaled, all_true)
    #solver options: saga, lbfgs

    #test logistic model
    save_output_test = SaveOutput()
    handle = model.module.fc6.register_forward_hook(save_output_test)
    all_true_test = []
    feature_set_test = []
    all_softmax_cnn = []
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader_test)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            all_true_test.extend(target.cpu().data.numpy())
            #all_pred_cnn.extend(np.argmax(output.cpu().data.numpy(), axis=1))
            m = torch.nn.Softmax(dim=0)
            for i,row in enumerate(output.cpu()):
                sm = m(row)
                all_softmax_cnn.append(sm.data.numpy())
    all_true_test = np.array(all_true_test)
    all_features_test = np.array(save_output_test.outputs)

    for item in all_features_test:
        feature_set_test.extend(item)
    all_features_test = np.array(feature_set_test)

    print(all_true_test.shape)
    print(all_features_test.shape)

    all_pred = clf.predict(minmaxscaler.transform(all_features_test))

    all_softmax_lr = clf.predict_proba(all_features_test)
    all_softmax_combined = all_softmax_lr + all_softmax_cnn
    all_pred_combined = np.argmax(all_softmax_combined, axis=1)

    #REASSIGN LABELS
    all_results = [all_pred, all_true_test, all_pred_combined]
    all_pred, all_true_test, all_pred_combined = reassign_labels(all_results, 0, 1)
    all_pred, all_true_test, all_pred_combined = reassign_labels(all_results, 9, 5)
    all_pred, all_true_test, all_pred_combined = reassign_labels(all_results, 10, 5)


    #display results
    util.plot_confusion_matrix(all_true_test, all_pred, classes=classes, normalize=False) #log regression
    util.plot_confusion_matrix(all_true_test, all_pred_combined, classes=classes, normalize=False) #average of log regression and cnn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()

    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume)