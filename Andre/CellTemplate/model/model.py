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
        super(threeDmodel_update,self).__init__()
        self.num_feature=num_feature
        
        self.conv_layer1 = self._make_conv_layer(1, self.num_feature)
        self.conv_layer2 = self._make_conv_layer(self.num_feature, self.num_feature*2)
        self.conv_layer3 = self._make_conv_layer(self.num_feature*2, self.num_feature*4)
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
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=1),
        nn.BatchNorm3d(out_c),
        nn.LeakyReLU(),
        nn.Conv3d(out_c, out_c, kernel_size=(3, 3, 3), padding=1),
        nn.BatchNorm3d(out_c),
        nn.LeakyReLU(),
        nn.MaxPool3d((mp_d, 2, 2)),
        )
        return conv_layer


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super(MnistModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    
class deeperModel(BaseModel):
    def __init__(self,num_feature=32, num_classes=2):
        super(deeperModel,self).__init__()
        self.num_feature=num_feature
        
        self.layer = nn.Sequential(
            nn.Conv2d(1,self.num_feature,5,1,2),
            nn.BatchNorm2d(self.num_feature),
            nn.ReLU(),
            nn.Conv2d(self.num_feature,self.num_feature*2,3,1,1),
            nn.BatchNorm2d(self.num_feature*2),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
            
            nn.Conv2d(self.num_feature*2,self.num_feature*4,3,1,1),
            nn.BatchNorm2d(self.num_feature*4),
            nn.ReLU(),
            nn.Conv2d(self.num_feature*4,self.num_feature*8,3,1,1),
            nn.BatchNorm2d(self.num_feature*8),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
            
            nn.Conv2d(self.num_feature*8,self.num_feature*16,3,1,1),
            nn.BatchNorm2d(self.num_feature*16),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
            nn.Conv2d(self.num_feature*16,self.num_feature*16,3,1,1),
            nn.BatchNorm2d(self.num_feature*16),
            nn.ReLU(),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(self.num_feature*16*3*3,1000),
            nn.ReLU(),
            nn.Linear(1000,num_classes)
        )       
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                
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
        out = self.layer(x)
        out = out.view(x.size()[0],-1)
        out = self.fc_layer(out)

        return out
    
class myCustomModel(BaseModel):
    def __init__(self, num_classes = 2):
        super(myCustomModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        return F.log_softmax(x, dim=1)
    
class pretrainedModel(BaseModel):
    def __init__(self):
        super(pretrainedModel, self).__init__()
        resnet = models.resnet152(pretrained=True)
        resnet.train()
        for param in resnet.parameters():
            param.requires_grad = False

        # new final layer with 10 classes
        num_ftrs = resnet.fc.in_features
        resnet.fc = torch.nn.Linear(num_ftrs, 2) #input is 224,224,3
            
        use_gpu = True
        if use_gpu:
            resnet = resnet.cuda()

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(resnet.fc.parameters(), lr=0.001, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        self.resnet = resnet
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Kaming Initialization
                init.kaiming_normal_(m.weight.data)
                #m.bias.data.fill_(0)
                init.normal_(m.bias.data)

    def forward(self, x):
        x = self.resnet(x)
        return x
    
class templateModel(BaseModel):
    def __init__(self):
        super(templateModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 43)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class groundTruthModel(BaseModel):
    def __init__(self, num_classes=2):
        super(groundTruthModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                
                # Xavier Initialization
                init.xavier_normal_(m.weight.data)
                init.normal_(m.bias.data)
                
            elif isinstance(m, nn.Linear):

                # Xavier Initialization
                init.xavier_normal_(m.weight.data)
                init.normal_(m.bias.data)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #print(x.shape)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    
    
class heatmapModel(BaseModel):
    def __init__(self,num_feature=32, num_classes=2):
        super(heatmapModel,self).__init__()
        self.num_feature=num_feature
        self.layer = nn.Sequential(
            nn.Conv2d(1,self.num_feature,3,1,1),
            nn.BatchNorm2d(self.num_feature),
            nn.ReLU(),
            nn.Conv2d(self.num_feature,self.num_feature*2,3,1,1),
            nn.BatchNorm2d(self.num_feature*2),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
            
            nn.Conv2d(self.num_feature*2,self.num_feature*4,3,1,1),
            nn.BatchNorm2d(self.num_feature*4),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
            
            nn.Conv2d(self.num_feature*4,self.num_feature*8,3,1,1),
            nn.BatchNorm2d(self.num_feature*8),
            nn.ReLU(),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(self.num_feature*8*7*7,self.num_feature*8),
            nn.ReLU(),
            nn.Linear(self.num_feature*8, num_classes)
        )    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
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
        print(x.shape)
        out = self.layer(x)
        print(out.shape)
        out = out.view(x.size()[0],-1)
        print(out.shape)
        out = self.fc_layer(out)
        return out

class heatmapModel64(BaseModel):
    def __init__(self,num_feature=32, num_classes=2):
        super(heatmapModel64,self).__init__()
        self.num_feature=num_feature
        self.layer = nn.Sequential(
            nn.Conv2d(1,self.num_feature,3,1,1),
            nn.BatchNorm2d(self.num_feature),
            nn.ReLU(),
            nn.Conv2d(self.num_feature,self.num_feature*2,3,1,1),
            nn.BatchNorm2d(self.num_feature*2),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
            
            nn.Conv2d(self.num_feature*2,self.num_feature*4,3,1,1),
            nn.BatchNorm2d(self.num_feature*4),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
            
            nn.Conv2d(self.num_feature*4,self.num_feature*8,3,1,1),
            nn.BatchNorm2d(self.num_feature*8),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
            
            nn.Conv2d(self.num_feature*8,self.num_feature*16,3,1,1),
            nn.BatchNorm2d(self.num_feature*16),
            nn.ReLU(),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(self.num_feature*16*8*8,self.num_feature*16),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.num_feature*16,num_classes)
        )    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
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
        out = self.layer(x)
        out = out.view(x.size()[0],-1)
        out = self.fc_layer(out)
        return out
    

class threeDmodel_simple(BaseModel):
    def __init__(self,num_feature=32, num_classes=2, depth=7):
        super(threeDmodel_simple,self).__init__()
        self.num_feature=num_feature
        
        self.conv_layer1 = self._make_conv_layer(1, self.num_feature)
        self.conv_layer2 = self._make_conv_layer(self.num_feature, self.num_feature*2)
        self.conv_layer4=nn.Conv3d(self.num_feature*2, self.num_feature*4, kernel_size=(1, 3, 3), padding=(0,1,1))
        
        self.fc5 = nn.Linear(self.num_feature*4*1*8*8, self.num_feature*4)
        self.relu = nn.LeakyReLU()
        self.batch0=nn.BatchNorm1d(self.num_feature*4)
        self.drop=nn.Dropout(p=0.5)
        self.fc6 = nn.Linear(self.num_feature*4, self.num_feature*2)
        self.relu1 = nn.LeakyReLU()
        self.batch1=nn.BatchNorm1d(self.num_feature*2)
        self.drop1=nn.Dropout(p=0.5)     
        self.fc7 = nn.Linear(self.num_feature*2, num_classes)
        
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
        x=self.conv_layer4(x)
        #print(x.size())
        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        x = self.relu(x)
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
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 5, 5), padding=(1,2,2)),
        nn.BatchNorm3d(out_c),
        nn.LeakyReLU(),
        nn.Conv3d(out_c, out_c, kernel_size=(3, 3, 3), padding=1),
        nn.BatchNorm3d(out_c),
        nn.LeakyReLU(),
        nn.MaxPool3d((mp_d, 2, 2)),
        )
        return conv_layer
