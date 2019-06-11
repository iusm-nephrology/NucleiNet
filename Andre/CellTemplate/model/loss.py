import torch.nn.functional as F
import torch.nn as nn

def nll_loss(weights):
    # model must have softmax layer 
    return nn.NLLLoss(weight = weights)

def cross_entropy_loss(weights):
    return nn.CrossEntropyLoss(weight = weights)

def bce_loss(weights):
    return nn.BCELoss(weight = weights)
