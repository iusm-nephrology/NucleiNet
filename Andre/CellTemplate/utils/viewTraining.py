import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torchvision

def graphLoss(saved_dir):
    config_saved_filename = os.path.join(saved_dir, "config.json")
    losses = []
    val_losses = []
    metric = []
    val_metric = []
    checkpoints = []
    '''
    for filename in os.listdir(saved_dir):
        name = os.path.join(saved_dir,filename) 
        if os.path.isfile(name) and name.endswith(".pth"):
            checkpoints.append(name)
        else:
            print("skipped file: " + filename)
    '''
    file_name = os.path.join(saved_dir, "model_best.pth")
    #file_name = os.path.join(saved_dir, "checkpoint-epoch100.pth")
    checkpoint = torch.load(file_name)
    logger = checkpoint['logger']
    for item in logger.entries:
        losses.append(logger.entries[item]['loss'])
        val_losses.append(logger.entries[item]['val_loss'])
        metric.append(logger.entries[item]['my_metric'])
        val_metric.append(logger.entries[item]['val_my_metric'])
    fig1 = plt.figure() 
    plt.plot(range(len(losses)), losses, 'r--', range(len(losses)), val_losses, 'b--')
    plt.title('Training and validation loss - red and blue respectively')
    plt.ylim(0,1.0)
    plt.show()
    
    fig2 = plt.figure()
    plt.plot(range(len(metric)), metric, 'r--', range(len(metric)), val_metric, 'b--')
    plt.title('Training and validation accruacy - red and blue respectively')
    plt.show()
    return {'loss': losses, 'val_loss': val_losses}
    
    
    