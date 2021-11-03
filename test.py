from CarlaEnvironment import CarlaEnvironment
import genetic
import numpy as np
import time
import sys
import cv2
from neuralNetwork import NeuralNetwork as NN
from neuralNetwork import TensorNeural as TN

def tensortest():
    tn = TN()

def main():
    cc = CarlaEnvironment(640, 480, True)
    cc.__del__()


if __name__ == '__main__':
    main()
    print("It works!")