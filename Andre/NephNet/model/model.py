import torch.nn as nn
import torch.nn.functional as F
import torch
from base import BaseModel
from torchvision import models
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.init as init


class threeDmodel(BaseModel):
    def __init__(self,num_feature=32, num_classes=2, depth=7):
        super(threeDmodel,self).__init__()
        self.num_feature=num_feature
        
        self.conv_layer1 = self._make_conv_layer(1, self.num_feature)
        self.conv_layer2 = self._make_conv_layer(self.num_feature, self.num_feature*2)
        self.conv_layer3 = self._make_conv_layer(self.num_feature*2, self.num_feature*4, mp_d=1)
        self.conv_layer4= nn.Sequential(
            nn.Conv3d(self.num_feature*4, self.num_feature*8, kernel_size=(1, 3, 3), padding=(0,1,1)),
            nn.BatchNorm3d(self.num_feature*8),
            nn.LeakyReLU())
        
        self.fc5 = nn.Linear(self.num_feature*8*1*4*4, self.num_feature*8)
        self.relu = nn.LeakyReLU()
        self.batch0=nn.BatchNorm1d(self.num_feature*8)
        self.drop=nn.Dropout(p=0.5)
        self.fc6 = nn.Linear(self.num_feature*8, self.num_feature*4)
        self.relu1 = nn.LeakyReLU()
        self.batch1=nn.BatchNorm1d(self.num_feature*4)
        self.drop1=nn.Dropout(p=0.5)     
        self.fc7 = nn.Linear(self.num_feature*4, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # Kaming Initialization
                init.kaiming_normal_(m.weight.data)
                #m.bias.data.fill_(0)
                init.normal_(m.bias.data)
            elif isinstance(m, nn.Linear):
                # Kaming Initialization
                init.kaiming_normal_(m.weight.data)
                #m.bias.data.fill_(0)
                init.normal_(m.bias.data)
                
        
    def forward(self,x):
        #print(x.size())
        x = self.conv_layer1(x)
        #print(x.size())
        x = self.conv_layer2(x)
        #print(x.size())
        x = self.conv_layer3(x)
        #print(x.size())
        x=self.conv_layer4(x)
        #print(x.size())
        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        x = self.relu(x)
        #print(x.size())
        x = self.batch0(x)
        x = self.drop(x)
        x = self.fc6(x)
        x = self.relu1(x)
        x = self.batch1(x)
        x = self.drop1(x)
        x1=x
        x = self.fc7(x)

        return x
    
    def _make_conv_layer(self, in_c, out_c, mp_d=2):
        # note that kernals in Conv3d are (depth, width, height)
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=(1,1,1)),
        nn.BatchNorm3d(out_c),
        nn.LeakyReLU(),
        nn.Conv3d(out_c, out_c, kernel_size=(3, 3, 3), padding=1),
        nn.BatchNorm3d(out_c),
        nn.LeakyReLU(),
        nn.MaxPool3d((mp_d, 2, 2)),
        )
        return conv_layer


class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_features):
        super(DenseBlock, self).__init__()
        self.relu = nn.LeakyReLU()
        self.bn = nn.BatchNorm3d(in_channels)
        self.conv1 = nn.Conv3d(in_channels, num_features, kernel_size=(3, 3, 3), padding=1)
        self.conv2 = nn.Conv3d(num_features, num_features, kernel_size=(3, 3, 3), padding=1)
        self.conv3 = nn.Conv3d(num_features*2, num_features, kernel_size=(3, 3, 3), padding=1)
        self.conv4 = nn.Conv3d(num_features*3, num_features, kernel_size=(3, 3, 3), padding=1)
        self.conv5 = nn.Conv3d(num_features*4, num_features, kernel_size=(3, 3, 3), padding=1)
    def forward(self, x):
        bn = self.bn(x)
        conv1 = self.relu(self.conv1(bn))
        conv2 = self.relu(self.conv2(conv1))
        temp = torch.cat([conv1,conv2],1)
        c2_dense = self.relu(temp)
        conv3 = self.relu(self.conv3(c2_dense))
        c3_dense = self.relu(torch.cat([conv1,conv2,conv3],1))
        conv4 = self.relu(self.conv4(c3_dense))
        c4_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4], 1))
        conv5 = self.relu(self.conv5(c4_dense))
        c5_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4,conv5],1))
        return c5_dense

class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_features):
        super(TransitionBlock, self).__init__()
        self.relu = nn.LeakyReLU()
        self.bn = nn.BatchNorm3d(out_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0)
        self.pool = nn.AvgPool3d(kernel_size = 2, stride = 2, padding = 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn(x)
        out = self.pool(x)
        return out

class Dense3D(BaseModel):
    def __init__(self,num_feature=16, num_classes=2):
        super(Dense3D,self).__init__()
        last_layer_features = num_feature*3 #maybe 96 is too few??
        self.llf = last_layer_features
        self.relu = nn.LeakyReLU()
        self.num_feature= num_feature
        self.low_conv = nn.Conv3d(in_channels = 1, out_channels = num_feature*2, kernel_size = 5, padding = 2)
        self.dense1 = self._makeDense(DenseBlock, num_feature*2, self.num_feature)
        self.dense2 = self._makeDense(DenseBlock, num_feature*4, self.num_feature)
        self.t1 = self._makeTransition(TransitionBlock, in_channels = num_feature*5, out_channels = num_feature*4)
        self.t2 = self._makeTransition(TransitionBlock, in_channels = num_feature*5, out_channels = last_layer_features)
        self.drop=nn.Dropout(p=0.5)
        self.bn = nn.BatchNorm3d(last_layer_features)
        self.pre_classifier = nn.Linear(last_layer_features*8*8*1, 512)
        self.classifier = nn.Linear(512, num_classes)
       
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # Kaming Initialization
                init.kaiming_normal_(m.weight.data)
                #m.bias.data.fill_(0)
                init.normal_(m.bias.data)
            elif isinstance(m, nn.Linear):
                # Kaming Initialization
                init.kaiming_normal_(m.weight.data)
                #m.bias.data.fill_(0)
                init.normal_(m.bias.data)
                
        
    def forward(self,x):
        x = self.relu(self.low_conv(x))
        x = self.dense1(x)
        #print(x.size())
        x = self.t1(x)
        #print(x.size())
        x = self.dense2(x)
        x = self.t2(x)
        #print(x.size())
        x = self.bn(x)
        #print(x.size())
        x = x.view(-1, self.llf*8*8*1)
        x = self.pre_classifier(x)
        x = self.drop(x)
        x = self.classifier(x)

        return x
    
    def _makeDense(self, block, in_channels, num_features):
        layers = []
        layers.append(block(in_channels, num_features))
        return  nn.Sequential(*layers)

    def _makeTransition(self, block, in_channels, out_channels):
        layers = []
        layers.append(block(in_channels, out_channels, self.num_feature))
        return  nn.Sequential(*layers)

