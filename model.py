import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict

# multi-task learning framework
class Net(nn.Module):
    def __init__(self, num_color_classes, num_cos_classes):
        super().__init__()
        self.net = models.resnet50(pretrained=True) # pretrained ResNet18 on ImageNet
        self.n_features = self.net.fc.in_features  # get the number of features in the last layer
        self.net.fc = nn.Identity() # replace the last layer without doing anything

        # classification head based on color
        self.net.fc1 = nn.Sequential(OrderedDict(
            [('linear', nn.Linear(self.n_features,self.n_features)),
            ('relu1', nn.ReLU()),
            ('linear', nn.Linear(self.n_features,self.n_features)),
            ('relu2', nn.ReLU()),
            ('final', nn.Linear(self.n_features, num_color_classes))]))
        
        # classification head based on clothes type
        self.net.fc2 = nn.Sequential(OrderedDict(
            [('linear', nn.Linear(self.n_features,self.n_features)),
            ('relu1', nn.ReLU()),
            ('linear', nn.Linear(self.n_features,self.n_features)),
            ('relu2', nn.ReLU()),
            ('final', nn.Linear(self.n_features, num_cos_classes))]))

        # freeze pretrained ResNet 
        for param in self.net.parameters():
            param.requires_grad = False 

        # train classification head
        for param in self.net.fc1.parameters():
            param.requires_grad = True
        for param in self.net.fc2.parameters():
            param.requires_grad = True


    def forward(self, x):
        features = self.net(x)
        color_head = self.net.fc1(features)
        cos_head = self.net.fc2(features)
        return color_head, cos_head