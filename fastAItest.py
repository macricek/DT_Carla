from fastai.vision.all import *
import matplotlib.pyplot as plt
import cv2
import numpy as np
import albumentations  as albu
import os
import torch
from fastseg import MobileV3Small
from fastAI import get_image_array_from_fn, label_func


def get_pred_for_mobilenet(model, img_array):
    with torch.no_grad():
        image_tensor = img_array.transpose(2, 0, 1).astype('float32') / 255
        x_tensor = torch.from_numpy(image_tensor).to("cuda").unsqueeze(0)
        model_output = F.softmax(model.forward(x_tensor), dim=1).cpu().numpy()
    return model_output

if __name__ == '__main__':
    torch.cuda.device(0)
    print(torch.cuda.get_device_name(0))

    DATA_DIR = "Kaggle/"

    x_train_dir = os.path.join(DATA_DIR, 'train')
    y_train_dir = os.path.join(DATA_DIR, 'train_label')

    x_valid_dir = os.path.join(DATA_DIR, 'val')
    y_valid_dir = os.path.join(DATA_DIR, 'val_label')

    img = cv2.imread(str(get_image_files(x_valid_dir)[3]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    learn = load_learner('seg.pkl')
    print(learn.model)

    plt.imshow(get_pred_for_mobilenet(learn.model.cuda(), img)[0][2])
    plt.show()


