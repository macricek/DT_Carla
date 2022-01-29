from fastai.vision.all import *
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import torch
from SegmentationAlbumentationsTransformation import SegmentationAlbumentationsTransformation
from fastseg import MobileV3Small


def get_image_array_from_fn(fn):
    image = cv2.imread(fn)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def label_func(fn):
    return str(fn).replace(".png", "_label.png").replace("train", "train_label").replace("val\\", "val_label\\")


if __name__ == '__main__':
    torch.cuda.device(0)
    print(torch.cuda.get_device_name(0))

    DATA_DIR = "../Kaggle/"

    x_train_dir = os.path.join(DATA_DIR, 'train')
    y_train_dir = os.path.join(DATA_DIR, 'train_label')

    x_valid_dir = os.path.join(DATA_DIR, 'val')
    y_valid_dir = os.path.join(DATA_DIR, 'val_label')

    my_get_image_files = partial(get_image_files, folders=["train", "val"])
    codes = np.array(['back', 'left', 'right'], dtype=str)
    carla = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
                    get_items = my_get_image_files,
                    get_y = label_func,
                    splitter = FuncSplitter(lambda x: str(x).find('validation_set') != -1),
                    item_tfms=[SegmentationAlbumentationsTransformation()])

    dls = carla.dataloaders(Path(DATA_DIR), path=Path("."), bs=2)
    dls.show_batch(max_n=6)
    plt.show()
    model = MobileV3Small(num_classes=3, use_aspp=True, num_filters=8)
    learn = Learner(dls, model, metrics=[DiceMulti()], cbs=ShowGraphCallback())
    learn.fine_tune(10)
    learn.export('seg_aug.pkl')
    torch.save(learn.model, './fastai_model_aug.pth')