import time

from fastai.vision.all import *
import matplotlib.pyplot as plt
import cv2
import numpy as np
import albumentations as albu
import os
import copy
import torch
from fastseg import MobileV3Small
from fastAI import get_image_array_from_fn, label_func

USE_AUG = True
pathToLearner = "seg.pkl"
if USE_AUG:
    pathToLearner.replace("seg", "seg_aug")


def predict(model, img_array):
    print("Start")
    start = time.time()
    with torch.no_grad():
        image_tensor = img_array.transpose(2, 0, 1).astype('float32') / 255
        x_tensor = torch.from_numpy(image_tensor).to("cuda").unsqueeze(0)
        model_output = F.softmax(model.forward(x_tensor), dim=1).cpu().numpy()
    end = time.time()
    s = end - start
    print(s)
    return model_output


if __name__ == '__main__':
    torch.cuda.device(0)
    DATA_DIR = "../Kaggle/"
    x_train_dir = os.path.join(DATA_DIR, 'train')
    y_train_dir = os.path.join(DATA_DIR, 'train_label')

    x_valid_dir = os.path.join(DATA_DIR, 'val')
    y_valid_dir = os.path.join(DATA_DIR, 'val_label')
    img = cv2.imread(str(get_image_files(x_valid_dir)[3]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    learn = load_learner(pathToLearner)
    back, left, right = predict(learn.model.cuda(), img)[0]

    def ld_detection_overlay(image, left_mask, right_mask):
        res = copy.copy(image)
        res[left_mask > 0.3, :] = [0, 0, 255]
        res[right_mask > 0.3, :] = [255, 0, 0]
        return res


    plt.imshow(ld_detection_overlay(img, left, right))
    plt.show()


    for i in range(0, 10):
        img = cv2.imread(str(get_image_files(x_valid_dir)[3]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ima = predict(learn.model.cuda(), img)[0][1]
    plt.imshow(ima)
    plt.show()

    for i in range(0, 10):
        start = time.time()
        img = cv2.imread(str(get_image_files(x_valid_dir)[3]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = learn.predict(img)
        ima = np.array(learn.predict(img)[0])
        end = time.time()
        s = end - start
        print(s)
    plt.imshow(ima)
    plt.show()




