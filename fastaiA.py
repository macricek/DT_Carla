from fastai.vision.all import *
import torch
import numpy as np
import os


def getLabelPath(imagePath):
    expected = ["train", "val"]
    stem = imagePath.stem + "_label"
    suffix = imagePath.suffix
    for e in expected:
        if e in str(imagePath):
            labelPath = str(imagePath).replace(e, e+"_label")
    fullLabel = labelPath.replace(imagePath.stem, stem)
    return fullLabel


def getImages(path1):
    return get_image_files(path1, folders=['train', 'val'])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.empty_cache()

path = 'Kaggle/'
print(os.listdir(path))
codes = np.array(['back', 'left', 'right'], dtype=str)

imageNames = getImages(path)
maskN = getLabelPath(imageNames[0])
print(maskN)

dataBlock = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
                   get_items = getImages,
                   get_y = getLabelPath,
                   splitter=FuncSplitter(lambda o: "val" in str(Path(o))),
                   batch_tfms=None)

data = dataBlock.dataloaders(Path(path), path=Path("."), bs=2)

data.show_batch(max_n=6)




# Let's do some data augmentation!
# We define a wrapper class to integrate albumentations library with fastai
# Read https://docs.fast.ai/tutorial.albumentations.html for details
class SegmentationAlbumentationsTransform(ItemTransform):
    split_idx = 0

    def __init__(self, aug): self.aug = aug

    def encodes(self, x):
        img, mask = x
        aug = self.aug(image=np.array(img), mask=np.array(mask))
        return PILImage.create(aug["image"]), PILMask.create(aug["mask"])



albu_transform_list = [
    albu.IAAAdditiveGaussianNoise(p=0.2),
    albu.OneOf(
        [
            albu.CLAHE(p=1),
            albu.RandomBrightness(p=1),
            albu.RandomGamma(p=1),
        ],
        p=0.6,
    ),
    albu.OneOf(
        [
            albu.IAASharpen(p=1),
            albu.Blur(blur_limit=3, p=1),
            albu.MotionBlur(blur_limit=3, p=1),
        ],
        p=0.6,
    ),
    albu.OneOf(
        [
            albu.RandomContrast(p=1),
            albu.HueSaturationValue(p=1),
        ],
        p=0.6,
    ),
]
albu_transform = albu.Compose(albu_transform_list)