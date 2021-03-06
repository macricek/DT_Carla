import enum

from CarlaEnvironment import CarlaEnvironment
import sys
from PyQt5.QtCore import QObject, QThread, QCoreApplication
import time
sys.path.insert(0, "fastAI")
from fastAI.fastAI import get_image_array_from_fn, label_func
from fastAI.CameraGeometry import CameraGeometry


class Results(enum.IntEnum):
    '''
    Enum of all current results
    '''
    none = -1
    All = 2
    Navigation = 5
    Lines_Navigation = 22
    Radar = 204
    Lines = 300
    Lines_Radar = 301

    def __str__(self):
        return self.value

    def title(self):
        return self.name.replace("_", ", ")


class Mode(enum.IntEnum):
    '''
    Enum of all current supported modes
    '''
    trainingDefault = 0
    showTrainingDefault = 1
    runTestDefault = 2
    trainingTraffic = 3
    showTrainingTraffic = 4
    runTestTraffic = 5
    runTestSecondPath = 6


class Main(QCoreApplication):
    '''
    Main Class. This should be turned on to see the functionality!
    @author: Marko Chylik
    @Date: May, 2022
    '''
    def __init__(self, data, mode):
        '''
        Start the program.
        :param data: which data will be used -> from Results enum
        :param mode: which mode we want -> from Mode enum
        '''
        super(Main, self).__init__([])
        self.time = time.time()
        self.data = data
        self.mode = mode

        self.carlaEnvironment = CarlaEnvironment(self, data=data.value, debug=False)

    def terminate(self):
        '''
        terminate the program
        :return: None
        '''
        print("Terminating MAIN!")
        try:
            self.carlaEnvironment.terminate()
        finally:
            sys.exit(0)

    def exec(self):
        '''
        Run defined mode by self.mode
        :return: exec method from QCoreApplication
        '''
        # DEFAULT scenarios
        if self.mode == Mode.trainingDefault:
            self.training(False)
        elif self.mode == Mode.showTrainingDefault:
            self.showTrainingResult(False)
        elif self.mode == Mode.runTestDefault:
            self.runTest(False, 1)
        # ADVANCED scenarios
        elif self.mode == Mode.trainingTraffic:
            self.training(True)
        elif self.mode == Mode.showTrainingTraffic:
            self.showTrainingResult(True)
        elif self.mode == Mode.runTestTraffic:
            self.runTest(True, 1)
        elif self.mode == Mode.runTestSecondPath:
            self.runTest(False, 2)

        return super().exec()

    def training(self, advanced):
        '''
        run training with traffic, if advanced
        :param advanced: bool
        :return: None
        '''
        self.carlaEnvironment.train(advanced)

    def showTrainingResult(self, advanced):
        '''
        Show how vehicle performed on training path.
        :param advanced: bool
        :return: None
        '''
        if self.data == Results.none:
            print("Need to pick some results")
            return
        self.carlaEnvironment.replayTrainingRide(data.value, advanced)

    def runTest(self, advanced, path):
        '''
        Run testing path
        :param advanced: bool
        :param path: 1 or 2
        :return: None
        '''
        if self.data == Results.none:
            print("Need to pick some results")
            return
        self.carlaEnvironment.testRide(data.value, advanced, path)


if __name__ == '__main__':
    '''
    Main code!
    '''
    data = Results.All                      # pick the data
    mode = Mode.showTrainingDefault         # pick the mode
    mainApp = Main(data, mode)              # start Main
    code = mainApp.exec()                   # run it
    sys.exit(code)                          # end the app with code
