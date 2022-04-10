import copy
import enum

import carla
import configparser
import queue
import os


def convertStringToBool(string: str):
    return string.lower() == "true"


class InputsEnum(enum.Enum):
    '''
    Inputs for neural network. Value will store number of inputs provided by each
    Inputs:
    Linedetect - 6 inputs, 3 points for left and right line detected by FastAI Detector
    Radar - 3 inputs, average of all valid scans in left (-45°, -25°), center (-10°, 10°) and right (25°, 45°) side
    Agent - 1 input, "suggested" steering
    Metrics - 2 inputs, current steering and steering 10 steps back
    BinaryKnowledge - 4 inputs, that will simply suggest where Car should turn [-1 for left, 1 for right, 0 for none]:
                    1: Based on lines detected (left line is closer than right -> turn right: 1)
                    2: Based on difference between agent steering and actual steering
                    3: Which radar measurement has the shortest range [-1 for left, 0 for center, 1 for right]
                    4: Based on actual speed towards goal
    Navigation - 2 inputs, errors toward waypoint in x, y
    '''

    linedetect = 6
    radar = 3
    agent = 1
    metrics = 2
    binaryknowledge = 4
    navigation = 0

    @staticmethod
    def elem(name):
        for element in InputsEnum:
            if name == element.name:
                return element

    def getValue(self):
        if self == InputsEnum.navigation:
            return 2
        else:
            return self.value


class CarlaConfig:
    client: carla.Client

    def __init__(self, client=None, path="config.ini"):
        self.client = client
        self.parser = configparser.ConfigParser()
        self.path = path
        self.parser.read(self.path)

    def apply(self):
        ''''APPLY CONFIG SETTINGS TO SIMULATOR'''
        self.sync = self.client.get_world().get_settings().synchronous_mode
        self.client.set_timeout(30)
        weather = self.parser.get("CARLA", "weather")
        map = self.parser.get("CARLA", "map")
        fixed = self.parser.get("CARLA", "ts")
        sync = self.parser.get("CARLA", "sync")
        world = self.client.get_world()
        curMap = world.get_map().name

        if map not in curMap:
            world = self.client.load_world(map)
        elif len(world.get_actors()) > 0:
            print("Re-load world!")
            world = self.client.reload_world()
        world.set_weather(eval(weather))

        if len(fixed) > 0:
            settings = world.get_settings()
            settings.fixed_delta_seconds = eval(fixed)
            world.apply_settings(settings)

        self.turnOnSync() if eval(sync) else self.turnOffSync()
        self.client.set_timeout(5)

    def readSection(self, section):
        return dict(self.parser.items(section))

    def loadNEData(self) -> dict:
        listIns, count = self.loadAskedInputs()
        self.parser.set("NE", "nInput", str(count))
        self.rewrite(self.path)
        expDict = self.readSection("NE")
        return expDict

    def rewrite(self, path):
        with open(path, 'w') as configfile:
            self.parser.write(configfile)

    def incrementNE(self):
        expDict = self.loadNEData()
        base = str(expDict.get("base"))

        if not os.path.exists(base):
            os.mkdir(base)
        old = int(self.parser.get("NE", "rev"))
        file = base + str(old) + "/"

        if not os.path.exists(file):
            os.mkdir(file)
        self.rewrite(file + "config.ini")

        self.parser.set("NE", "rev", str(old + 1))
        self.rewrite(self.path)

    def loadAskedInputs(self) -> (list, int):
        nnIns = self.readSection("NnInputs")
        expectedList = []
        count = 0

        for key, val in nnIns.items():
            if convertStringToBool(val):
                inputElement = InputsEnum.elem(key)
                expectedList.append(inputElement)
                count += inputElement.getValue()

        return expectedList, count

    def turnOffSync(self):
        world = self.client.get_world()
        if self.sync:
            settings = self.client.get_world().get_settings()
            settings.synchronous_mode = False
            self.sync = False
            world.apply_settings(settings)
            print("Sync mode turned off!")

    def turnOnSync(self):
        world = self.client.get_world()
        if not self.sync:
            settings = self.client.get_world().get_settings()
            settings.synchronous_mode = True
            self.sync = True
            world.apply_settings(settings)
            print("Sync mode turned on!")

    @staticmethod
    def loadPath() -> queue.Queue:
        path = queue.Queue()
        x = [-8.6, -121.8, -363.4, -463.6, -490]
        y = [107.5, 395.2, 406, 333.6, 174]
        for i in range(len(x)):
            loc = carla.Location()
            loc.x = x[i]
            loc.y = y[i]
            loc.z = 0.267
            path.put(loc)
        return path


if __name__ == '__main__':
    conf = CarlaConfig()
    s = conf.readSection("NE")
    print(s)
    from neuralNetwork import NeuralNetwork as NN
    nn = NN()
