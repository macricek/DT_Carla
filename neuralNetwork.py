import math
from collections import deque

import numpy as np


def loadNNParamsFromConfig(nnConfig):
    '''
    Load Neural Network parameters -> size of input, hidden, outputs
    :param nnConfig: config.ini parser
    :return: size of inputs, hidden, outputs in expected shape for NN
    '''
    nInputs = int(nnConfig.get("ninput"))
    nHiddenOne = int(nnConfig.get("nhidden"))
    nHidden = []
    hiddenLayers = int(nnConfig.get("nhiddenlayers"))
    for _ in range(hiddenLayers):
        nHidden.append(nHiddenOne)
    nHidden = np.asarray(nHidden)
    nOutputs = int(nnConfig.get("noutput"))

    return nInputs, nHidden, nOutputs


def check(num, limit):
    '''
    Check if num is in range <-limit,limit>. If not, return max/min
    :param num: number
    :param limit: upper range
    :return: checked number
    '''
    if num > limit:
        return limit
    elif num < -limit:
        return -limit
    else:
        return num


class NeuralNetwork:
    '''
    Basic MLP neural network implementation
    @author: Marko Chylik
    @author: Marko Chylik
    @Date: May, 2022
    '''
    nInput = 0
    nHiddenLayers = 0
    nHidden = None
    nOutput = 0
    szWeights = 0
    szBiases = 0
    weights = None

    def __init__(self, nInput, nHiddenInL, nOutput, weights=None):
        '''
        Constructor of NN.
        :param nInput: number of inputs [INT]
        :param nHiddenInL: number of hidden neurons [np.array]
        :param nOutput: number of outputs [INT]
        :param weights: array of weights -> using method setWeights to update weights.
        '''
        self.nInput = nInput
        self.nHiddenLayers = nHiddenInL.shape[0]
        self.nHidden = nHiddenInL
        self.nOutput = nOutput
        self.szWeights, self.szBiases = self.getNumOfNeededElements()
        self.weights = weights

    def setWeights(self, weights):
        '''
        Update weights to new value
        :param weights: np.array of correct size
        :return: Nothing
        '''
        assert weights.shape[0] == self.szWeights + self.szBiases
        self.weights = weights

    def getNumOfNeededElements(self):
        '''
        Calculate number of needed weights and biases to fill in NN.
        :return: size of weights, biases [INT, INT]
        '''
        szW = self.nInput * self.nHidden[0] + self.nOutput * self.nHidden[1]
        szB = self.nOutput + self.nHidden[-1]
        for i in range(0, self.nHiddenLayers - 1):
            szW += self.nHidden[i] * self.nHidden[i + 1]
            szB += self.nHidden[i]
        return szW, szB

    def parse(self):
        '''
        Parse self.weights array to more arrays that'll be used in NN calculations
        :return: Weights, Biases
        '''
        assert self.weights.shape[0] == self.szWeights + self.szBiases
        # weights index
        idxW1end = self.nInput * self.nHidden[0]
        idxW2start = idxW1end
        idxW2end = idxW2start + self.nHidden[0] * self.nHidden[1]
        idxW3start = idxW2end
        idxW3end = idxW3start + self.nHidden[1] * self.nOutput
        # biases index
        idxBH1start = idxW3end
        idxBH1end = idxBH1start + self.nHidden[0]
        idxBH2start = idxBH1end
        idxBH2end = idxBH2start + self.nHidden[1]
        idxBOstart = idxBH2end
        idxBOend = idxBOstart + self.nOutput
        v = self.weights.shape[0]
        assert idxBOend == self.weights.shape[0]
        # weights parsing
        W1 = np.reshape(self.weights[0:idxW1end], (self.nInput, self.nHidden[0]))
        W2 = np.reshape(self.weights[idxW2start:idxW2end], (self.nHidden[0], self.nHidden[1]))
        W3 = np.reshape(self.weights[idxW3start:idxW3end], (self.nHidden[1], self.nOutput))
        # biases parsing
        BH1 = self.weights[idxBH1start:idxBH1end]
        BH2 = self.weights[idxBH2start:idxBH2end]
        BO = self.weights[idxBOstart:idxBOend]
        return W1, W2, W3, BH1, BH2, BO

    def run(self, inputs, limit):
        '''
        Run one step of NN
        :param inputs: array of inputs [-1;1]
        :param limit: coef to outputs
        :return: outputs array
        '''
        assert inputs.shape[0] >= self.nInput
        if inputs.shape[0] > self.nInput:
            X = inputs[0:self.nInput]
        else:
            X = inputs
        W1, W2, W3, BH1, BH2, BO = self.parse()

        H1 = np.zeros((1, self.nHidden[0]))
        H2 = np.zeros((1, self.nHidden[1]))
        O = np.zeros((1, self.nOutput))

        tmp = X @ W1 + BH1
        for i in range(0, self.nHidden[0]):
            H1[0, i] = math.tanh(tmp[i])

        tmp = H1 @ W2 + BH2
        for i in range(0, self.nHidden[1]):
            H2[0, i] = math.tanh(tmp[0, i])

        tmp = H2 @ W3 + BO
        for i in range(0, self.nOutput):
            O[0, i] = math.tanh(tmp[0, i])

        return O * limit

    @staticmethod
    def normalizeLinesInputs(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        '''
        Normalize lines to MLP range
        :param left: numpy array with points of left line
        :param right: numpy array with points of right line
        :return: normalized numpy array of both lines together
        '''

        leftRight = np.concatenate((left, right), axis=0)
        norm = np.linalg.norm(leftRight)
        if norm == 0:
            return leftRight
        return leftRight / norm

    @staticmethod
    def normalizeRadarInputs(radar: np.ndarray) -> np.ndarray:
        '''
        Normalize radar measures, as we expect that maximal distance of measurement is 50
        :param radar: measures
        :return: 0 - 1 range of measures
        '''

        coef = 50
        if np.max(radar) > coef:
            coef = np.max(radar)
        return 1 - (radar / coef)

    @staticmethod
    def normalizeAgent(agent: int) -> np.ndarray:
        '''
        Transform agent to ndarray
        :param agent: agent's suggested steering
        :return: ndarray (1x1) for NN
        '''
        return np.array([agent])

    @staticmethod
    def normalizeMetrics(metrics: deque, limit) -> np.ndarray:
        '''
        Normalize metrics by current vehicle's steering limit
        :param metrics: deque consists of two values
        :param limit: steering limit of vehicle
        :return: ndarray (1x2) for NN
        '''
        mList = []
        if len(metrics) < 2:
            return np.array([0, 0])

        for m in metrics:
            mList.append(check(m.steer, limit))

        return np.asarray(mList) / limit

    @staticmethod
    def normalizeBinary(binary: list) -> np.ndarray:
        '''
        Convert binary inputs from list to ndarray
        :param binary: list of binary inputs
        :return: ndarray (1x4) for NN
        '''
        return np.asarray(binary)

    @staticmethod
    def normalizeNavigation(currentLocation, waypoint) -> np.ndarray:
        '''
        normalizing into {-1,1}. Expecting max will be 2 metres away.
        :param currentLocation: carla.Location of vehicle
        :param waypoint: carla.Location of wp
        :return: ndarray (1x2) for NN
        '''
        if not waypoint:
            return np.asarray([0, 0])

        xErr = (currentLocation.x - waypoint.x) / 2
        yErr = (currentLocation.y - waypoint.y) / 2

        return np.asarray([check(xErr, 1), check(yErr, 1)])
