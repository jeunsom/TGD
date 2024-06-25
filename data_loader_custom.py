import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import numpy as np

from torch.autograd import Variable as V

import os
import random
import copy
import csv
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import numpy as np

NUM_WORKERS = 4

class Cifar10(Dataset):
  def __init__(self, csv_file_path = './data/cifar10/train.csv', 
               file_pathR='./data/cifar10/train_PIr', 
               file_pathC = './data/cifar10/train_PIc', 
               file_pathOrg = './data/cifar10/train',
               augment6 = None, 
               augment3 = None, 
               class_to_idx = {'airplane': 0,
                               'automobile': 1,
                               'bird': 2,
                               'cat': 3,
                               'deer': 4,
                               'dog': 5,
                               'frog': 6,
                               'horse': 7,
                               'ship': 8,
                               'truck': 9}, 
               shuffle=True):

    self.img_listR = []
    self.img_listC = []
    self.img_listOrg = [] 
    self.img_label = []
    self.augment6 = augment6
    self.augment3 = augment3
    self.class_to_idx = class_to_idx

    with open(csv_file_path, "r") as fileDescriptor:
      csvReader = csv.reader(fileDescriptor)
      next(csvReader, None)
      for line in csvReader:
        imagePathR = os.path.join(file_pathR, line[0])
        imagePathC = os.path.join(file_pathC, line[0])
        imagePathOrg = os.path.join(file_pathOrg, line[0])
        label = class_to_idx[line[1]]
        self.img_listR.append(imagePathR)
        self.img_listC.append(imagePathC)
        self.img_listOrg.append(imagePathOrg)
        self.img_label.append(label)

    indexes = np.arange(len(self.img_listR))

    if shuffle:
      np.random.shuffle(indexes)
      _img_listR, _img_listC, _img_listOrg, _img_label = copy.deepcopy(self.img_listR), copy.deepcopy(self.img_listC), copy.deepcopy(self.img_listOrg), copy.deepcopy(self.img_label)
      self.img_listR = []
      self.img_listC = []
      self.img_listOrg = []
      self.img_label = []

      for i in indexes:
        self.img_listR.append(_img_listR[i])
        self.img_listC.append(_img_listC[i])
        self.img_listOrg.append(_img_listOrg[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):

    imagePathR = self.img_listR[index]
    imagePathC = self.img_listC[index]
    imagePathOrg = self.img_listOrg[index]

    imageDataR = Image.open(imagePathR).convert('RGB')
    imageDataC = Image.open(imagePathC).convert('RGB')
    stu_imageData = np.concatenate((imageDataR, imageDataC), axis=2)
    tea_imageData = Image.open(imagePathOrg).convert('RGB')
    imageLabel = self.img_label[index]

    if self.augment6 != None:
      stu_imageData, tea_imageData = self.augment6(stu_imageData), self.augment3(tea_imageData)


    return stu_imageData, tea_imageData, imageLabel

  def __len__(self):

    return len(self.img_listR)

def get_cifar(dataset_dir='./Data/cifar10', batch_size=128, crop=False):
    normalizeT = transforms.Normalize(mean=[0.05165074, 0.05487815, 0.06697452, 0.05165074, 0.05487815, 0.06697452], std=[0.12468914, 0.12733056, 0.13768335, 0.12468914, 0.12733056, 0.13768335])
    simple_transform6 = transforms.Compose([transforms.ToTensor(), normalizeT])
    normalizeS = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    simple_transform3 = transforms.Compose([transforms.ToTensor(), normalizeS])
    if crop is True:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ])
    else:
        train_transform = simple_transform6
   
    trainset = Cifar10(csv_file_path = os.path.join(dataset_dir, 'train.csv'),
                       file_pathR = os.path.join(dataset_dir, 'train_PIr'),
                       file_pathC = os.path.join(dataset_dir, 'train_PIc'),
                       file_pathOrg = os.path.join(dataset_dir, 'train'),
                       augment6 = simple_transform6,
                       augment3 = simple_transform3)	
    testset = Cifar10(csv_file_path = os.path.join(dataset_dir, 'test.csv'),
                       file_pathR = os.path.join(dataset_dir, 'test_PIr'),
                       file_pathC = os.path.join(dataset_dir, 'test_PIc'),
                       file_pathOrg = os.path.join(dataset_dir, 'test'),
                       augment6 = simple_transform6,
                       augment3 = simple_transform3)	
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=NUM_WORKERS,
											  pin_memory=True, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=NUM_WORKERS,
											 pin_memory=True, shuffle=False)
    return trainloader, testloader

