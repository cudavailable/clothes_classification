import torch
import os
import json
import torch.nn as nn
import torch.optim as optim

from model import Net
from train import train_epoch, test_model
from data_processing import make_train_test_sample, train_test_split, get_train_test_data, get_label
from google.colab import drive

# Code from google colab
try:
    from google.colab import drive
    drive.mount('/content/drive')
    workspace = '/content/drive/MyDrive/Colab Notebooks'
except:
    workspace = '.'


def main():
    # Load meta.json file
    root = os.path.join(workspace, "clothes")
    meta_path = os.path.join(root, "meta.json")
    meta = json.load(open(meta_path,'r'))
    images_info = meta['images']

    # Prepare datasets for training & testing
    img_sets = make_train_test_sample(root, images_info)
    samples = train_test_split(root, img_sets, images_info)
    train_imgs, test_imgs = get_train_test_data(samples)

    y_color_train, y_cos_train, y_color_test, y_cos_test = get_label(samples)

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model
    model = Net(num_color_classes=6, num_cos_classes=5)
    model.to(device)

    # Loss function
    color_criterion = nn.CrossEntropyLoss()
    cos_criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    # Model training
    train_epoch(model,  train_imgs, (y_color_train, y_cos_train), 
                (color_criterion, cos_criterion), optimizer)
    
    # Model testing
    test_model(model, test_imgs, (y_color_test, y_cos_test))
