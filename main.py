from CarlaEnvironment import CarlaEnvironment
import sys
from PyQt5.QtCore import QObject, QThread, QCoreApplication
import time
sys.path.insert(0, "fastAI")
from fastAI.fastAI import get_image_array_from_fn, label_func
from fastAI.CameraGeometry import CameraGeometry


class Main(QCoreApplication):
    def __init__(self):
        super(Main, self).__init__([])
        self.time = time.time()
        self.carlaEnvironment = CarlaEnvironment(self)  # , data=103, debug=False)
        self.carlaEnvironment.train()
        # self.carlaEnvironment.testRide(102)

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
