import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict


# 设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 多任务学习框架
class Net(nn.Module):
  def __init__(self, num_color_classes, num_cos_classes):
      super().__init__()
      self.net = models.resnet18(pretrained=True) # 使用在ImageNet上预训练权重的ResNet18
      self.n_features = self.net.fc.in_features  # 得到最后一层的输入特征个数
      self.net.fc = nn.Identity() # 将最后一层替换，不做任何操作

      self.net.fc1 = nn.Sequential(OrderedDict(
          [('linear', nn.Linear(self.n_features,self.n_features)),
          ('relu1', nn.ReLU()),
          ('final', nn.Linear(self.n_features, num_color_classes))])) # 根据衣物颜色进行分类

      self.net.fc2 = nn.Sequential(OrderedDict(
          [('linear', nn.Linear(self.n_features,self.n_features)),
          ('relu1', nn.ReLU()),
          ('final', nn.Linear(self.n_features, num_cos_classes))]))  # 根据衣物种类进行分类

      # 冻结 ResNet 除了最后一层的权重参数
      for param in self.net.parameters():
          param.requires_grad = False  # 先将所有参数设置为不需要梯度

      # 仅允许最后的全连接层（自定义部分）进行训练
      for param in self.net.fc1.parameters():
          param.requires_grad = True
      for param in self.net.fc2.parameters():
          param.requires_grad = True


  def forward(self, x):
      features = self.net(x)
      color_head = self.net.fc1(features)
      cos_head = self.net.fc2(features)
      return color_head, cos_head
