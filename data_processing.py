import os
import random
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from sample import Sample

def list_dir_in_directory(directory):
    
    dirs = os.listdir(directory)
    d_list = [d for d in dirs if os.path.isdir(os.path.join(directory, d))]

    # Return a list containing names of every directory
    return d_list

def make_train_test_sample(root, images_info):
    """Get training & testing samples (included in img_sets)"""

    # Create a empty list for each class in the directory
    d_list = list_dir_in_directory(root)
    img_sets = {}
    for d in d_list:
      img_sets[d] = []

    for i, img_info in enumerate(images_info):
      # path = img_info['path']
      cls = img_info['class']
      # label = img_info['label']

      cls = cls[0] + "_" + cls[1]
      img_sets[cls].append(i) # Append index into specific list according to its class

    # Pick 20 elements randomly from each class(training + testing)
    for cls in img_sets.keys():
      temp = random.sample(img_sets[cls], 20)
      img_sets[cls].clear()
      img_sets[cls] = temp

    return img_sets

def train_test_split(root, img_sets, images_info):
    """Split training & testing samples according to img_sets"""

    samples = {'train': [], 'val': []}

    # Split training[:17] & testing[17:] datasets
    for cls in img_sets.keys():
        smp_list = img_sets[cls]
        for i, idx in enumerate(smp_list):
            img_info = images_info[idx]

            fpath = os.path.join(root, img_info['path'])
            if not os.path.isfile(fpath):
                raise ValueError('%s not found' % fpath)
            else:
                img = cv2.imread(fpath, cv2.IMREAD_COLOR)[..., ::-1] # BGR to RGB
            if len(samples['train']) == 0 and len(samples['val'])==0:
                H, W, C = img.shape
            else:
                cv2.resize(img, (W, H))

            if i < 17:
                samples['train'].append(Sample(img=img, classPair=img_info['class'], labelList=img_info['label']))
            else:
                samples['val'].append(Sample(img=img, classPair=img_info['class'], labelList=img_info['label']))

    return samples

def img_transform(img_rgb, transform=None):
  """
  transform images
  :param img_rgb: PIL Image
  :param transform: torchvision.transform
  :return: tensor
  """
    
  if transform is None:
    raise ValueError("there is no transform")
    
  img_t = transform(Image.fromarray(img_rgb))
  return img_t

def get_train_test_data(samples):
    """Get training & testing data"""

    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    inference_transform = transforms.Compose([
      transforms.Resize((256, 256)),
      transforms.ToTensor(),
      transforms.Normalize(norm_mean, norm_std),
    ])

    # Load training & testing data
    train_imgs = [img_transform(sample.img, inference_transform) for sample in samples['train']]
    train_imgs = torch.stack(train_imgs, dim=0)

    test_imgs = [img_transform(sample.img, inference_transform) for sample in samples['val']]
    test_imgs = torch.stack(test_imgs, dim=0)

    return train_imgs, test_imgs

def get_label(samples):
    """Get y labels"""
    # Labels for training
    color_label = [sample.labelColor for sample in samples['train']]
    cos_label = [sample.labelCos for sample in samples['train']]

    # Transfer one hot vector to class no.
    y_color_train = torch.argmax(torch.tensor(color_label), dim=1).to(torch.long)
    y_cos_train = torch.argmax(torch.tensor(cos_label), dim=1).to(torch.long)

    # print(y_color_train.shape)
    # print(y_cos_train.shape)
    
    # Labels for testing
    y_color_test = np.array([sample.labelColor for sample in samples['val']])
    y_color_test = np.argmax(y_color_test, axis=1)
    y_cos_test = np.array([sample.labelCos for sample in samples['val']])
    y_cos_test = np.argmax(y_cos_test, axis=1)

    return y_color_train, y_cos_train, y_color_test, y_cos_test