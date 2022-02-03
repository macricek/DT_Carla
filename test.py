import LineDetection
from CarlaEnvironment import CarlaEnvironment
import numpy as np
import cv2
from LineDetection import CNNLineDetector, transformImage
import torch
import os
import matplotlib.pyplot as plt
import random
from fastai.vision.all import *
import sys
sys.path.insert(0, "fastAI")
from fastAI import get_image_array_from_fn, label_func
import time

USE_AUG = True
pathToLearner = "fastAI\\seg.pkl"
if USE_AUG:
    pathToLearner.replace("seg", "seg_aug")

def lineDetectorPredict(im, cnnD):
    shapeIm = np.shape(im)
    transformedImage, _ = transformImage(im, LineDetection.testtransform, np.empty(shapeIm))
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
    cnnD = CNNLineDetector(False, dataPath=LineDetection.data_path)
    learn = load_learner(pathToLearner)
    fig, axs = plt.subplots(numImages, 4, figsize=(10, 5 * numImages))
    for i in range(numImages):
        rand = int(random.random() * sizeOf)
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
        maskLineDetector = lineDetectorPredict(img, cnnD)
        #predict with fastAI
        maskFastAI = np.array(learn.predict(imgT)[0])
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

    plt.show()
    fig.savefig('figs\\Compare.png')


def main():
    try:
        cc = CarlaEnvironment(1, True)
        cc.run()

    finally:
        print("Invoking deletion of object Environment")
        cc.deleteAll()
        del cc


if __name__ == '__main__':
    #torch.cuda.device(0)
    #DATA_DIR = "Kaggle/"
    #x_train_dir = os.path.join(DATA_DIR, 'train')
    #y_train_dir = os.path.join(DATA_DIR, 'train_label')

    #x_valid_dir = os.path.join(DATA_DIR, 'val')
    #y_valid_dir = os.path.join(DATA_DIR, 'val_label')
    #sizeOf = os.listdir(x_valid_dir).__len__()
    #lineD()
    # #compare(5)
    main()
