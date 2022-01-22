import LineDetection
from CarlaEnvironment import CarlaEnvironment
import numpy as np
import cv2
from LineDetection import CNNLineDetector, transformImage
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import time

def lineD():
    im = cv2.imread("_out/00048254.png")
    shapeIm = np.shape(im)
    transformedImage,_ = transformImage(im, LineDetection.testtransform, np.empty(shapeIm))
    cnnD = CNNLineDetector(False, dataPath=LineDetection.data_path)
    start = time.time()
    mask = cnnD.predict(transformedImage.squeeze())
    im1 = mask.cpu()
    end = time.time()
    s = end - start
    print(s)
    vis = im1.numpy()
    shapee = np.shape(vis)
    vis2 = np.reshape(vis, (shapee[0], shapee[1], 1))
    vis2 = vis2.astype('uint8')
    norm = np.linalg.norm(vis2)
    normal_array = vis2 / norm
    vis2 = normal_array * 255
    cv2.imshow("Image", vis2)
    cv2.waitKey()


def main():
    try:
        cc = CarlaEnvironment(1, True)
        cc.run()

    finally:
        print("Invoking deletion of object Environment")
        cc.deleteAll()
        del cc


if __name__ == '__main__':
    lineD()
    #main()