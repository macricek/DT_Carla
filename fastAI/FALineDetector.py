import copy
import os
import torch
import warnings
import cv2
import numpy as np

from fastAI import *
from fastai.vision.all import *
from fastAI.CameraGeometry import CameraGeometry
#from CameraGeometry import CameraGeometry


class FALineDetector:
    torchModel: str
    fastAiModel: str
    image: ndarray
    left: ndarray
    right: ndarray
    mask: ndarray

    def __init__(self, aug=True, isMain=False):
        self.importModels(aug, isMain)
        self.device = "cuda"
        self.learner = load_learner(self.fastAiModel)
        self.model = torch.load(self.torchModel).to(self.device)
        self.time = time.time()
        self.treshold = 0.3
        self.cg = CameraGeometry()
        self.cut_v, self.grid = self.cg.precompute_grid()
        self.init(isMain)
        warnings.filterwarnings("error")

    def importModels(self, aug, ismain):
        if ismain:
            self.torchModel = os.path.abspath('fastai_model.pth')
            self.fastAiModel = os.path.abspath('seg.pkl')
        else:
            self.torchModel = os.path.join('fastAI/fastai_model.pth')
            self.fastAiModel = os.path.join('fastAI/seg.pkl')
        if aug:
            self.torchModel = self.torchModel.replace("model", "model_aug")
            self.fastAiModel = self.fastAiModel.replace("seg", "seg_aug")

    # First detection takes 2-3s so make "fake" on beginning
    def init(self, isMain):
        self.loadImage(path="im.png" if isMain else "fastAI/im.png")
        self.predict()
        self.model.eval()

    def predict(self):
        with torch.no_grad():
            image_tensor = self.image.transpose(2, 0, 1).astype('float32') / 255
            x_tensor = torch.from_numpy(image_tensor).to("cuda").unsqueeze(0)
            _, self.left, self.right = F.softmax(self.model.forward(x_tensor), dim=1).cpu().numpy()[0]

        self.left = self.filter(self.left, 5)
        self.right = self.filter(self.right, 5)

    def filter(self, inputImage, it=1) -> np.ndarray:
        kernel = np.ones((5, 5), np.uint8)
        retVal = inputImage
        dilated = cv2.dilate(retVal, kernel, iterations=it)
        eroded = cv2.erode(dilated, kernel, iterations=it)
        retVal = eroded
        return retVal

    def integrateLines(self):
        self.image[self.left > self.treshold, :] = [0, 0, 255]  # blue for left
        self.image[self.right > self.treshold, :] = [255, 0, 0]  # red for right

    def visualize(self, delay=1):
        self.integrateLines()
        cv2.imshow("Visualization of LineDetection", self.image)
        cv2.waitKey(delay)

    def visualizeLines(self):
        cv2.imshow("Left Line", self.left)
        cv2.imshow("Right Line", self.right)
        cv2.waitKey()

    def extractPolynomials(self):
        try:
            leftPolynomial = self.fit(self.left)
            rightPolynomial = self.fit(self.right)
        # in specific conditions, LineDetector is not relevant, so we rather not use this detected lines
        # numpy will return Warning, which we change to an Error to be able catching it.
        except:
            leftPolynomial = np.poly1d(np.array([0., 0., 0., 0.]))
            rightPolynomial = np.poly1d(np.array([0., 0., 0., 0.]))
        return leftPolynomial, rightPolynomial

    def fit(self, probs):
        probs_flat = np.ravel(probs[self.cut_v:, :])
        mask = probs_flat > 0.3
        if mask.sum() > 0:
            coeffs = np.polyfit(self.grid[:, 0][mask], self.grid[:, 1][mask], deg=3, w=probs_flat[mask])
        else:
            coeffs = np.array([0., 0., 0., 0.])
        return np.poly1d(coeffs)

    def loadImage(self, path=None, numpyArr=None):
        if path is not None:
            self.image = cv2.imread(path)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        else:
            self.image = numpyArr

    def sinceLast(self):
        since = time.time() - self.time
        self.time = time.time()
        return since


if __name__ == '__main__':
    fald = FALineDetector(aug=True, isMain=True)
    d = os.path.join("../Kaggle/val/")
    for file in os.listdir("../Kaggle/val"):
        fileP = os.path.join(d + file)
        print(fileP)
        print(fald.sinceLast())
        fald.loadImage(path=fileP)
        fald.predict()
        fald.visualize(10)
        left, right = fald.extractPolynomials()
        x = np.linspace(0, 30, num=6)
        yl = left(x)
        yr = right(x)
        plt.plot(x, yl, label="yl")
        plt.plot(x, yr, label="yr")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.legend()
        plt.axis("equal")
        plt.show()
        #time.sleep(0.5)
