import enum

from CarlaEnvironment import CarlaEnvironment
import sys
from PyQt5.QtCore import QObject, QThread, QCoreApplication
import time
sys.path.insert(0, "fastAI")
from fastAI.fastAI import get_image_array_from_fn, label_func
from fastAI.CameraGeometry import CameraGeometry


class Results(enum.IntEnum):
    none = -1
    lines = 61
    lines_radar_agent = 103
    lines_measure = 201


class Main(QCoreApplication):
    def __init__(self):
        super(Main, self).__init__([])
        self.time = time.time()
        data = Results.lines_measure

        self.carlaEnvironment = CarlaEnvironment(self, data=data.value, debug=False)
        if data == Results.none:
            self.carlaEnvironment.train()
        else:
            self.carlaEnvironment.testRide(data.value)

    def terminate(self):
        print("Terminating MAIN!")
        try:
            self.carlaEnvironment.terminate()
        finally:
            sys.exit(0)

    def signal_handler(self, sig, frame):
        print('You pressed Ctrl+C!')
        sys.exit(0)


if __name__ == '__main__':
    mainApp = Main()
    sys.exit(mainApp.exec())
