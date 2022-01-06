import LineDetection
from CarlaEnvironment import CarlaEnvironment
import numpy as np
import cv2
from LineDetection import CNNLineDetector, transformImage
import albumentations as A
from albumentations.pytorch import ToTensorV2


def lineD():
    im = cv2.imread("Kaggle/val/Town04_Clear_Noon_09_09_2020_14_57_22_frame_1_validation_set.png")
    shapeIm = np.shape(im)
    transformedImage = transformImage(im, LineDetection.testtransform, np.empty(shapeIm))


def main():
    try:
        cc = CarlaEnvironment(1, True)
        cc.run()

    finally:
        print("Invoking deletion of object Environment")
        cc.deleteAll()
        del cc


if __name__ == '__main__':
    #lineD()
    main()