import os
import argparse
import torch
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
import json
from utils import util
import pandas as pd
import matplotlib.pyplot as plt

def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def set_instance(module, name, config, *args):
    setattr(module, config[name]['type'])(*args, **config[name]['args'])

def interogate_CM(all_pred, all_true, true_class, pred_class, dataloader, classes):
    #Comparison variables
    CM_images = []

    true_class_bool = np.array(all_true) == true_class
    pred_class_bool = np.array(all_pred) == pred_class
    comparison_bool = np.logical_and(true_class_bool, pred_class_bool)
    comparison_idx = [i for i, val in enumerate(comparison_bool) if val]
    for idx in comparison_idx:
        num_dim = np.squeeze(dataloader.dataset.__getitem__(idx)[0].data.numpy()).ndim
        if num_dim > 2:
            CM_images.append(np.sum(np.squeeze(dataloader.dataset.__getitem__(idx)[0].data.numpy()), axis=0))
        else:
            CM_images.append(np.squeeze(dataloader.dataset.__getitem__(idx)[0].data.numpy()))

    pl = 10
    pw = 10
    title = "True class: {}, Pred class: {}".format(classes[true_class], classes[pred_class])
    fig, axs = plt.subplots(pl, pw, facecolor = 'w', edgecolor = 'k')
    plt.suptitle((title), fontsize = 10, fontweight = 'bold')
    axs = axs.ravel()
    fig.subplots_adjust(hspace = 0.05, wspace = 0.05)
    for i in range(pw*pl):
        if i < len(CM_images):
            axs[i].imshow(CM_images[i], cmap = plt.cm.gray)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    plt.show()

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




    #ADJUST CD45 LABELS
    for i, pred in enumerate(all_pred_k):
        for j,p in enumerate(pred):
            if p >= 9:
                all_pred_k[i, j] = 5 #change all the prediction to CD45 class
            elif p == 0:
                all_pred_k[i,j] = 1 #change all prediction of S1 to S2
    ###############
    pct_mask = all_true_k == 0
    all_true_k[pct_mask] = 1 #change ground truth to remove S1 in favor of general PCT class


   
    
def main(config, resume, config_testdata = None):

    #PREFERENCES
    outputOverlaycsv = False
    showHeatMap = False
    THRESHOLD = 0.75
    TRUE_CLASS = 7
    PRED_CLASS = 5
    
    # set visualization preference
    print("GPUs available: " + str(torch.cuda.device_count()))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    # setup data_loader instances
    data_loader = get_instance(module_data, 'data_loader_test', config_testdata)
    '''
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )
    '''

    # build model architecture
    #config['arch']['args']['num_feature'] = 76
    model = get_instance(module_arch, 'arch', config)
    print(model)
    if torch.cuda.is_available():
        print("Using GPU: " + torch.cuda.get_device_name(0))
    else:
        print("Using CPU to test")
        
    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    criterion = loss_fn(None) # for imbalanced datasets
    #criterion = loss_fn(data_loader.dataset.weight.to(device)) # for imbalanced datasets
    
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # load state dict
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    
    #classes = ('endothelium', 'pct', 'vasculature')
    classes = ('s1', 's2s3', 'tal', 'dct', 'cd', 'cd45', 'nestin', 'cd31_glom', 'cd31_inter', 'cd45_1', 'cd45_2')
    all_pred = []
    all_pred_k = []
    all_true = []
    all_softmax = []

    thresh_true = []
    thresh_pred = []

    thresh_true_bad = []
    thresh_pred_bad = []
    below_thresh = 0
    

    
    if showHeatMap:
        hm_layers = {'final_layer': 'layer', 'fc_layer': 'fc_layer', 'conv_num': 17, 'fc_num': 3} #need to set based on model
        heatmapper = classActivationMap.CAMgenerator(hm_layers, config, model)
        #heatmapper = classActivationMap.CAMgenerator3d(hm_layers, config, model)  #for 3d data
        heatmapper.generateImage(num_images=10)
    
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            k=2
            data, target = data.to(device), target.to(device)
            output = model(data)
            image = np.squeeze(data[0].cpu().data.numpy())
            label = np.squeeze(target[0].cpu().data.numpy())
            all_true.extend(target.cpu().data.numpy())
            all_pred.extend(np.argmax(output.cpu().data.numpy(), axis=1))
            mypred = torch.topk(output, k, dim=1)[1]
            all_pred_k.extend(mypred.cpu().data.numpy())
            m = torch.nn.Softmax(dim=0)
            for i,row in enumerate(output.cpu()):
                sm = m(row)
                all_softmax.append(sm.data.numpy())
                #print(np.max(sm.numpy(), axis=0))
                if np.amax(sm.numpy(), axis=0) > THRESHOLD:
                    thresh_true.append(target.cpu().data.numpy()[i])
                    thresh_pred.append(np.argmax(sm))
                else:
                    below_thresh +=1
                    thresh_true_bad.append(target.cpu().data.numpy()[i])
                    thresh_pred_bad.append(np.argmax(sm))

            if i < 2:
                m = torch.nn.Softmax(dim=0)
                print("prediction percentages")
                print(m(output.cpu()[0]))
                print(all_true[i])
                #all_softmax.extend(m(output.cpu()))
            #if i > 50: break
            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            loss = criterion(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output.cpu(), target.cpu()) * batch_size
            
    correct = 0
    all_pred_k = np.array(all_pred_k)
    all_true_k = np.array(all_true)   
 
    #CREATE 8 CLASS SOFTMAX
    #very ugly please fix
    softmax_combined = np.zeros((len(all_softmax), len(all_softmax[0]) - 3))
    for i, row in enumerate(all_softmax):
        max_pct = np.amax([row[0], row[1]])
        max_cd45 = np.amax([row[5], row[9], row[10]])
        myrow = row[1:len(all_softmax[0]) - 3+1]
        myrow[0] = max_pct
        myrow[4] = max_cd45
        softmax_combined[i] = myrow

    all_pred_k_combined = torch.topk(torch.from_numpy(softmax_combined), k, dim=1)[1]+1

    all_pred_k = torch.from_numpy(all_pred_k)
    all_true_k = torch.from_numpy(np.array(all_true))
    for i in range(k):
        correct += torch.sum(all_pred_k[:,i] == all_true_k).item()
    print("TOP K all classes = {}".format(correct / len(all_pred_k)))


    all_pred_k = np.array(all_pred_k)
    all_true_k = np.array(all_true)   


    #REASSIGN LABELS
    all_results = [all_pred, all_true, thresh_true, thresh_pred, thresh_true_bad, thresh_pred_bad, all_pred_k, all_true_k]
    all_pred, all_true, thresh_true, thresh_pred, thresh_true_bad, thresh_pred_bad, all_pred_k, all_true_k = reassign_labels(all_results, 0, 1)
    all_pred, all_true, thresh_true, thresh_pred, thresh_true_bad, thresh_pred_bad, all_pred_k, all_true_k = reassign_labels(all_results, 9, 5)
    all_pred, all_true, thresh_true, thresh_pred, thresh_true_bad, thresh_pred_bad, all_pred_k, all_true_k = reassign_labels(all_results, 10, 5)

    print(np.unique(all_true))
    print(np.unique(all_pred))
    #VIEW MISTAKE CLASS
    interogate_CM(all_pred, all_true, TRUE_CLASS, PRED_CLASS, data_loader, classes)

    correct = 0
    all_pred_k = torch.from_numpy(all_pred_k)
    all_true_k = torch.from_numpy(np.array(all_true))
    for i in range(k):
        correct += torch.sum(all_pred_k_combined[:,i] == all_true_k).item()
    print("TOP K Combined classes = {}".format(correct / len(all_pred_k_combined)))


    if outputOverlaycsv:
        ids = data_loader.dataset.getIds()
        #softmax = pd.DataFrame(all_softmax)
        softmax = pd.DataFrame(softmax_combined)
        #ids = ids[:,1].reshape(ids.shape[0], 1)
        print(ids[0:5])
        print(ids.shape)
        print(softmax.shape)
        frames = [ids, softmax, pd.DataFrame(all_true)]
        output_data= np.concatenate(frames, axis=1)
        print(output_data.shape)
        output_df = pd.DataFrame(output_data)
        output_df.to_csv('overlaycsv.csv', index=False,  header=False)
        
    n_samples = len(data_loader.sampler)
    print("num test images = " + str(n_samples))
    log = {'loss': total_loss / n_samples}
    log.update({met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)})
    for key in log:
        print("{} = {:.4f}".format(key, log[key]))
    #print(log)
    log['classes'] = classes
    log['test_targets'] = all_true
    log['test_predictions'] = all_pred
    print("My_metric is accuracy")
    print("Number of images below threshold: {}".format(below_thresh))
    util.plot_confusion_matrix(all_true, all_pred, classes=classes, normalize=False)
    util.plot_confusion_matrix(thresh_true, thresh_pred, classes=classes, normalize=False)
    util.plot_confusion_matrix(thresh_true_bad, thresh_pred_bad, classes=classes, normalize=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default = None, type = str, help = 'path to config for testing')
    parser.add_argument('-t', '--test', default = None, type = str, help = 'path to h5 for testing')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()

    if args.resume:
        config = torch.load(args.resume)['config']
        config_testdata = config
    if args.config:
        with open(args.config) as handle:
            config_testdata = json.load(handle)
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    if args.test:
        config_testdata['data_loader_test']['args']['hdf5_path'] = args.test
        config_testdata['data_loader_test']['args']['mean'] = 16.37,
        config_testdata['data_loader_test']['args']['stdev'] = 19.74



    main(config, args.resume, config_testdata)
