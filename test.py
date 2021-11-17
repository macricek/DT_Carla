from CarlaEnvironment import CarlaEnvironment
import genetic
import numpy as np
import time
import sys
import cv2
from neuralNetwork import NeuralNetwork as NN


def main():
    try:
        cc = CarlaEnvironment(2, True)
        cc.run()

    finally:
        print("Invoking deletion of object Environment")
        cc.deleteAll()
        del cc


if __name__ == '__main__':
    main()