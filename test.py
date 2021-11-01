from carlaCar import CarlaCar
import genetic
import numpy as np
import time
import sys
import cv2
from neuralNetwork import NeuralNetwork as NN


def main():
    cc = CarlaCar(640, 480, True)

    #cv2.imshow("ukazka", cc.frontView)
    #cv2.waitKey(0)
    cc.__del__()


if __name__ == '__main__':
    main()
    print("It works!")