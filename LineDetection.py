import os
import glob
from pathlib import Path
import time
import math
from random import randint
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mtc
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from sklearn.model_selection import KFold
import seaborn as sns
import segmentation_models_pytorch as smp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def sample(dataset, num_samples):
    samples = []
    for i in range(num_samples):
        n = randint(0, len(dataset))
        img, mask = dataset[n]
        samples.append((img.squeeze(), mask.squeeze()))
    return samples


def showDataset(dataset, num_imgs):
    imgs = sample(dataset, num_imgs)
    fig, axs = plt.subplots(num_imgs, 2, figsize=(10, 5 * num_imgs))
    for i in range(num_imgs):
        # Image
        axs[i, 0].imshow(imgs[i][0].permute(1, 2, 0))

        # Original Mask
        axs[i, 1].imshow(imgs[i][1])
    plt.show()


class LaneDetectionDataset(Dataset):
    def __init__(self, path, val=False, transforms=None):
        self.transforms = transforms
        if not val:
            self.img_path = path + 'train/'
            self.mask_path = path + 'train_label/'
        else:
            self.img_path = path + 'val/'
            self.mask_path = path + 'val_label/'
        self.img_names = [name for name in os.listdir(self.img_path)]

    def __getitem__(self, idx, imShow=False):
        img_name = self.img_names[idx]
        mask_name = img_name[:-4] + '_label' + img_name[-4:]
        img = cv2.imread(self.img_path + img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path + mask_name, cv2.IMREAD_UNCHANGED)
        if imShow:
            cv2.imshow("Image", img)
            cv2.imshow("Mask", mask)
            cv2.waitKey(0)
        if self.transforms:
            transformed = self.transforms(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask'].long()
        return img, mask

    def __len__(self):
        return len(self.img_names)


class DeepNeuralNetwork:
    def __init__(self, from_scratch=True, path='unet_model.pth', best=False):
        self.val_epoch = None
        self.train_epoch = None
        self.optimizer = None
        self.loss = None

        self.train_dataset = LaneDetectionDataset(data_path, val=False, transforms=testtransform)
        self.val_dataset = LaneDetectionDataset(data_path, val=True, transforms=testtransform)
        self.trainloader = DataLoader(self.train_dataset, batch_size=4, shuffle=True)
        self.valloader = DataLoader(self.val_dataset, batch_size=4, shuffle=True)
        self.path = path

        self.model = smp.UnetPlusPlus(encoder_name='resnet34',
                                      encoder_weights='imagenet',
                                      in_channels=3,
                                      classes=3).to(device)
        if from_scratch:
            print('Model initialised from Scratch.')

        elif best:
            path = path.split('.')
            path[0] += '_best'
            path = '.'.join(path)
            self.model.load_state_dict(torch.load(path))
            print('Loaded saved model at: ', path)
        else:
            self.model.load_state_dict(torch.load(path))
            print('Loaded saved model at: ', path)

    def setTrainingParams(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.loss = smp.losses.DiceLoss('multiclass')
        self.loss.__name__ = 'DiceLoss'

        self.train_epoch = smp.utils.train.TrainEpoch(
            self.model,
            loss=self.loss,
            metrics=[],
            optimizer=self.optimizer,
            device=device,
            verbose=True
        )

        self.val_epoch = smp.utils.train.ValidEpoch(
            self.model,
            loss=self.loss,
            metrics=[],
            device=device,
            verbose=True
        )

    def train(self, num_epochs, save=True):
        for i in range(1, num_epochs+1):
            print("Running epoch {now}/{max}".format(now=i, max=num_epochs))
            logTraining = self.train_epoch.run(self.trainloader)
            logValidation = self.val_epoch.run(self.valloader)
            print("TRAINING STATUS")
            print(logTraining, logValidation)
        if save:
            torch.save(self.model.cpu().state_dict(), self.path)
            print("Model saved to " + self.path)


# Global Definitions
model_path = 'unet_model.pth'
data_path = 'Kaggle/'

# Augmentation
testtransform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Define datasets
train_dataset = LaneDetectionDataset(data_path, val=False, transforms=testtransform)
val_dataset = LaneDetectionDataset(data_path, val=True, transforms=testtransform)

showDataset(num_imgs=4, dataset=val_dataset)

model = DeepNeuralNetwork(from_scratch=True)
model.setTrainingParams()
model.train(num_epochs=2, save=True)

