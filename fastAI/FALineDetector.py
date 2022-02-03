import copy
import os
import time

from fastai.vision.all import *
import torch
import cv2
import numpy as np
from fastAI import *


class FALineDetector:
    torchModel: str
    fastAiModel: str
    image: ndarray
    left: ndarray
    right: ndarray
    mask: ndarray

    def __init__(self, aug=True, learner=False):
        self.importModels(aug)
        self.useLearner = learner
        self.device = torch.cuda.device(0)
        self.learner = load_learner(self.fastAiModel)
        self.model = self.learner.model.cuda()
        self.time = time.time()
        self.initLearners()

    def importModels(self, aug):
        self.torchModel = os.path.abspath('fastai_model.pth')
        self.fastAiModel = os.path.abspath('seg.pkl')
        if aug:
            self.torchModel = self.torchModel.replace("model", "model_aug")
            self.fastAiModel = self.fastAiModel.replace("seg", "seg_aug")

    def predictByLearner(self):
        self.mask = np.array(self.learner.predict(self.image)[0])

    def predict(self):
        with torch.no_grad():
            image_tensor = self.image.transpose(2, 0, 1).astype('float32') / 255
            x_tensor = torch.from_numpy(image_tensor).to("cuda").unsqueeze(0)
            _, self.left, self.right = F.softmax(self.model.forward(x_tensor), dim=1).cpu().numpy()[0]

    def visualize(self):
        actImage = self.image
        if self.useLearner:
            actImage[self.model > 0.1, :] = [0, 255, 0]  # use green separators
        else:
            actImage[self.left > 0.1, :] = [0, 0, 255]  # blue for left
            actImage[self.right > 0.1, :] = [255, 0, 0]  # red for right
        cv2.imshow("Visualization of LineDetection", actImage)
        cv2.waitKey(1)

    def loadImage(self, path=None, numpyArr=None):
        if path is not None:
            self.image = cv2.imread(path)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        else:
            self.image = numpyArr

    # First detection takes 2-3s so make "fake" on beginning
    def initLearners(self):
        self.loadImage(path="im.png")
        self.predictByLearner() if self.useLearner else self.predict()

    def sinceLast(self):
        since = time.time() - self.time
        self.time = time.time()
        return since


if __name__ == '__main__':
    fald = FALineDetector(aug=True, learner=False)
    d = os.path.join("../Kaggle/val/")
    for file in os.listdir("../Kaggle/val"):
        fileP = os.path.join(d + file)
        print(fileP)
        print(fald.sinceLast())
        fald.loadImage(path=fileP)
        fald.predict()
        fald.visualize()
        #time.sleep(0.5)
