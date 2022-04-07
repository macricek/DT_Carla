import torch
from CarlaEnvironment import CarlaEnvironment
import numpy as np
import cv2
import sys
sys.path.insert(0, "torchUnet")
from torchUnet.UnetLineDetection import *
import os
import matplotlib.pyplot as plt
import random
from fastai.vision.all import *

sys.path.insert(0, "fastAI")
from fastAI.fastAI import *
import time
import signal
import sys

USE_AUG = True
pathToLearner = "fastAI\\seg.pkl"
if USE_AUG:
    pathToLearner.replace("seg", "seg_aug")


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)


def lineDetectorPredict(im, cnnD):
    shapeIm = np.shape(im)
    transformedImage, _ = transformImage(im, testtransform, np.empty(shapeIm))
    mask = cnnD.predict(transformedImage.squeeze())
    im1 = mask.cpu()
    vis = im1.numpy()
    shapee = np.shape(vis)
    vis2 = np.reshape(vis, (shapee[0], shapee[1], 1))
    vis2 = vis2.astype('uint8')
    norm = np.linalg.norm(vis2)
    normal_array = vis2 / norm
    vis2 = normal_array * 255
    return vis2


def compare(numImages):
    cnnD = CNNLineDetector(False, dataPath=data_path)
    learn = load_learner(pathToLearner)
    fig, axs = plt.subplots(numImages, 4, figsize=(10, 5 * numImages))
    unet = []
    fi = []
    for i in range(numImages):
        rand = int(random.random() * 100)
        #Image
        img = cv2.imread(str(get_image_files(x_valid_dir)[rand]))
        imgT = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #Mask
        mask = cv2.imread(str(get_image_files(y_valid_dir)[rand]))
        mask = mask.astype('uint8')
        norm = np.linalg.norm(mask)
        normal_array = mask / norm
        mask = normal_array * 255
        #predict with CNNLineDetector
        t1 = time.time()
        maskLineDetector = lineDetectorPredict(img, cnnD)
        unet.append(time.time() - t1)
        #predict with fastAI
        t2 = time.time()
        maskFastAI = np.array(learn.predict(imgT)[0])
        fi.append(time.time() - t2)
        #AXS
        axs[i, 0].imshow(imgT)
        axs[i, 1].imshow(mask)
        axs[i, 2].imshow(maskLineDetector)
        axs[i, 3].imshow(maskFastAI)
        if i == 0:
            axs[i, 0].set_title("Image")
            axs[i, 1].set_title("Mask")
            axs[i, 2].set_title("ResNet34")
            axs[i, 3].set_title("FastAI")
        axs[i, 0].axis('off')
        axs[i, 1].axis('off')
        axs[i, 2].axis('off')
        axs[i, 3].axis('off')
    unet = np.asarray(unet[1:10])
    fi = np.asarray(fi[1:10])

    plt.show()
    fig.savefig('figs\\Compare.png')
    return unet, fi


def times(unet, fi):
    plt.plot(unet, '*', label='Unet++')
    plt.plot(fi, 'o', label='FastAI')
    plt.title("Porovnanie rýchlosti segmentácie")
    plt.xlabel("Poradové číslo segmentácie")
    plt.ylabel("Čas segmentácie [s]")
    plt.legend()
    plt.show()

def pyplot():
    arr = []
    for _ in range(10):
        arr.append(10000)
    plt.plot(np.asarray(arr))
    plt.show()


if __name__ == '__main__':
    unet, fi = compare(11)
    times(unet, fi)
