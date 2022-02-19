from fastai.vision.all import *
import numpy as np
import albumentations as albu


# Let's do some data augmentation!
# We define a wrapper class to integrate albumentations library with fastai
# Read https://docs.fast.ai/tutorial.albumentations.html for details
class SegmentationAlbumentationsTransformation(ItemTransform):
    split_idx = 0

    def __init__(self, transform=None):
        if transform is None:
            self.aug = self.useDefaultTransform()
        else:
            self.aug = transform

    def encodes(self, x):
        img, mask = x
        aug = self.aug(image=np.array(img), mask=np.array(mask))
        return PILImage.create(aug["image"]), PILMask.create(aug["mask"])

    @staticmethod
    def useDefaultTransform():
        albuList = [albu.IAAAdditiveGaussianNoise(p=0.2),
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
        transform = albu.Compose(albuList)
        return transform
