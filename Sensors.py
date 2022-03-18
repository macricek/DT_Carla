import copy
from statistics import fmean
import threading

import carla
import math
import cv2
import numpy as np
from carla import ColorConverter as cc
import queue
from PyQt5 import QtCore


def convertStringToBool(string: str):
    return string.lower() == "true"


class SensorManager(QtCore.QObject):
    """
    In case of Sync, we need to manage sensors when server ticks.
    If mode is async, we could simply rely on callbacks.
    """

    def __init__(self, vehicle, environment):
        super().__init__()
        self._queues = []
        self.sensors = []
        self.cameras = []
        self.count = 0
        self.readySensors = 0
        self.vehicle = vehicle

        self.environment = environment
        self.config = environment.config
        self.debug = not environment.trainingMode

        self.ldCam = LineDetectorCamera(self, False, False)
        self.rgbCam = Camera(self, False, False)
        self.segCam = SegmentationCamera(self, False, False)

        self.collision = CollisionSensor(self, False)
        self.radar = RadarSensor(self, False)
        self.lidar = LidarSensor(self, False)
        self.obstacleDetector = ObstacleDetector(self, False)
        self.laneInvasionDetector = LaneInvasionDetector(self, False)

        self.addToSensorsList()
        self.activate()

    def addToSensorsList(self):
        settings: dict
        settings = self.config.readSection("Sensors")
        # SENSORS
        if convertStringToBool(settings.get("radarsensor")):
            self.sensors.append(self.radar)
        if convertStringToBool(settings.get("collisionsensor")):
            self.sensors.append(self.collision)
        if convertStringToBool(settings.get("obstacledetector")):
            self.sensors.append(self.obstacleDetector)
        if convertStringToBool(settings.get("lidarsensor")):
            self.sensors.append(self.lidar)
        if convertStringToBool(settings.get("laneinvasiondetector")):
            self.sensors.append(self.laneInvasionDetector)
        # CAMERAS
        if convertStringToBool(settings.get("linedetectorcamera")):
            self.sensors.append(self.ldCam)
            self.cameras.append(self.ldCam)
        if convertStringToBool(settings.get("defaultcamera")):
            self.sensors.append(self.rgbCam)
            self.cameras.append(self.rgbCam)
        if convertStringToBool(settings.get("segmentationcamera")):
            self.sensors.append(self.segCam)
            self.cameras.append(self.segCam)

    def activate(self):
        for sensor in self.sensors:
            self.print(f"Activating {self.count}: {sensor.name}")
            self.count += 1
            sensor.activate()

    def processSensors(self):
        for sensor in self.sensors:
            sensor.on_world_tick()

    def isCollided(self):
        return self.collision.isCollided()

    def lines(self) -> (np.ndarray, np.ndarray):
        retLeft = copy.deepcopy(self.ldCam.left)
        retRight = copy.deepcopy(self.ldCam.right)
        self.ldCam.resetLines()
        return retLeft, retRight

    def radarMeasurement(self) -> np.ndarray:
        return self.radar.returnAverageRanges()

    def destroy(self):
        self.print(f"Invoking deletion of sensors of {self.vehicle.vehicleID} vehicle!")
        for sensor in self.sensors:
            self.print(f"Deleting sensor {sensor.name}")
            sensor.destroy()

    def print(self, message):
        if self.debug:
            print(message)

    def applyTesting(self):
        self.debug = True
        self.rgbCam.show = True
        # TODO: idk if processor is overhelmed by many cameras or what
        # for camera in self.cameras:
        #     camera.show = True


class Sensor(QtCore.QObject):
    debug: bool = False

    def __init__(self, manager, debug):
        super(Sensor, self).__init__()
        self.sensor = None
        self.bp = None
        self.where = None
        self.manager = manager
        self.name = "Sensor"
        # self.ready = False
        self.queue = queue.Queue()
        self.vehicle = manager.vehicle
        self.debug = debug

    def callBack(self, data):
        pass

    def activate(self):
        self.sensor = self.world().spawn_actor(self.bp, self.where, attach_to=self.reference())
        self.sensor.listen(lambda data: self.queue.put(data))

    def on_world_tick(self):
        if self.debug:
            print(f"[{self.name}] on world tick")
        if self.queue.qsize() > 0:
            self.callBack(self.queue.get())
        # print(f"Emitting ready for sensor {self.name}")

    def reference(self):
        return self.vehicle.ref()

    def setVehicle(self, vehicle):
        self.vehicle = vehicle

    def blueprints(self):
        return self.vehicle.environment.blueprints

    def world(self):
        return self.vehicle.environment.world

    def lineDetector(self):
        return self.vehicle.fald

    def config(self):
        return self.vehicle.environment.config

    def destroy(self):
        if self.sensor is not None:
            try:
                self.sensor.destroy()
            except:
                pass


class RadarSensor(Sensor):
    def __init__(self, manager, debug=False):
        super().__init__(manager, debug)
        self.range = 50
        self.name = "Radar"
        self.bp = self.blueprints().find('sensor.other.radar')
        self.bp.set_attribute('horizontal_fov', str(90))
        self.bp.set_attribute('vertical_fov', str(25))
        self.where = carla.Transform(carla.Location(x=1.5, z=0.5))

        atDeg = [-35, 0, 35]
        mR = 10

        self.detectedFinal = []

        self.leftRange = range(atDeg[0] - mR, atDeg[0] + mR)
        self.centerRange = range(atDeg[1] - mR, atDeg[1] + mR)
        self.rightRange = range(atDeg[2] - mR, atDeg[2] + mR)

        self.left = 0
        self.right = 0
        self.center = 0

    def callBack(self, data):
        left = []
        right = []
        center = []

        for detect in data:
            azi = int(math.degrees(detect.azimuth))
            alt = math.degrees(detect.altitude)
            dist = detect.depth
            if alt > 0:
                if azi in self.leftRange:
                    left.append(dist)
                elif azi in self.centerRange:
                    center.append(dist)
                elif azi in self.rightRange:
                    right.append(dist)

        self.left = fmean(left) if len(left) > 0 else self.range
        self.right = fmean(right) if len(right) > 0 else self.range
        self.center = fmean(center) if len(center) > 0 else self.range

    def returnAverageRanges(self) -> np.ndarray:
        return np.array([self.left, self.center, self.right])


class CollisionSensor(Sensor):
    collided: bool

    def __init__(self, manager, debug=False):
        super().__init__(manager, debug)
        self.name = "Collision"
        self.collided = False
        self.bp = self.blueprints().find('sensor.other.collision')
        self.where = carla.Transform(carla.Location(x=1.5, z=0.7))

    def callBack(self, data):
        self.collided = True
        print("Vehicle {id} collided!".format(id=self.vehicle.vehicleID))

    def isCollided(self):
        return self.collided


class ObstacleDetector(Sensor):
    def __init__(self, vehicle, debug=False):
        super().__init__(vehicle, debug)
        self.name = "Obstacle"
        self.bp = self.blueprints().find('sensor.other.obstacle')
        self.where = carla.Transform(carla.Location(x=1.5, z=0.7))

    def callBack(self, data):
        distance = data.distance
        print("{distance}".format(distance=distance))


class LidarSensor(Sensor):
    def __init__(self, manager, debug=False):
        super().__init__(manager, debug)
        self.name = "Lidar"
        self.bp = self.blueprints().find('sensor.lidar.ray_cast')
        self.bp.channels = 1
        self.where = carla.Transform(carla.Location(x=0, y=0, z=0))

    def callBack(self, data):
        if self.debug:
            number = 0
            for location in data:
                print("{num}: {location}".format(num=number, location=location))
                number += 1


class LaneInvasionDetector(Sensor):
    def __init__(self, manager, debug=False):
        super().__init__(manager, debug)
        self.name = "LaneInvasion"
        self.bp = self.blueprints().find('sensor.other.lane_invasion')
        self.where = carla.Transform(carla.Location(x=0, y=0, z=0))

        self.crossings = 0
        self.lastCross = -5

    def callBack(self, data):
        frameCrossed = data.frame
        if self.debug:
            print(f"Vehicle {self.vehicle.vehicleID} crossed line!")
            print(f"Crossed at frame {frameCrossed}, last cross was at {self.lastCross}")
        self.crossings += 1
        self.lastCross = frameCrossed


class Camera(Sensor):
    name: str

    options = {
        'Main': ['sensor.camera.rgb', cc.Raw],
        'Depth': ['sensor.camera.depth', cc.LogarithmicDepth],
        'Semantic Segmentation': ['sensor.camera.semantic_segmentation', cc.CityScapesPalette],
        'LineDetection': ['sensor.camera.rgb', cc.Raw]
    }

    def __init__(self, manager, debug=False, show=True):
        super().__init__(manager, debug)
        d = self.config().readSection('Camera')
        self.camHeight = int(d["height"])
        self.camWidth = int(d["width"])
        self.image = None
        self.show = show
        self.drawingThread = None
        self.stop = False
        self.bound_x = 0.5 + self.reference().bounding_box.extent.x
        self.bound_y = 0.5 + self.reference().bounding_box.extent.y
        self.bound_z = 0.5 + self.reference().bounding_box.extent.z
        self.name = 'Main'
        self.create()

    def create(self):
        typeOfCamera = self.options.get(self.name)[0]
        self.bp = self.blueprints().find(typeOfCamera)
        self.bp.set_attribute('image_size_x', f'{self.camWidth}')
        self.bp.set_attribute('image_size_y', f'{self.camHeight}')
        self.bp.set_attribute('fov', '110')
        self.where = carla.Transform(carla.Location(x=-2.0 * self.bound_x, y=+0.0 * self.bound_y, z=2.0 * self.bound_z),
                                     carla.Rotation(pitch=-8.0))

    def callBack(self, data):
        if self.debug:
            print(f"Entering {self.name} callback!")
        data.convert(self.options.get(self.name)[1])
        i = np.array(data.raw_data)
        i2 = i.reshape((self.camHeight, self.camWidth, 4))
        self.image = i2[:, :, :3]
        if self.isMain():
            self.invokeDraw()

    def draw(self):
        '''
        IN SEPARATE THREAD!
        :return: Nothing
        '''
        print(f"[{self.name}]Starting drawing process")
        while True:
            drawingImg = copy.copy(self.image)
            if self.stop:
                print("STOP")
                self.stop = False
                break
            cv2.imshow("Vehicle {id}, Camera {n}".format(id=self.vehicle.vehicleID, n=self.name), drawingImg)
            cv2.waitKey(1)

    def invokeDraw(self):
        if self.show and self.drawingThread is None:
            self.drawingThread = threading.Thread(target=self.draw)
            self.drawingThread.start()

    def isMain(self):
        return self.name == 'Main'

    def destroy(self):
        if self.drawingThread is not None:
            self.show = False
            self.stop = True  # break showing threads
            for _ in range(10):
                if self.stop:
                    self.manager.environment.tick()
            self.drawingThread = None
        super(Camera, self).destroy()


class LineDetectorCamera(Camera):
    left: np.ndarray
    right: np.ndarray

    def __init__(self, manager, debug=False, show=True):
        super(LineDetectorCamera, self).__init__(manager, debug, show)
        self.name = 'LineDetection'
        self.create()
        self.at = np.array([0, 7.5, 15])
        self.resetLines()

    def predict(self):
        '''
        loads image to LineDetector and predicts where are lines
        :return: Nothing
        '''
        self.lineDetector().loadImage(numpyArr=self.image)
        self.lineDetector().predict()

    def lines(self):
        '''
        Left and right points in range "self.at"
        :return:
        '''
        ll, rl = self.lineDetector().extractPolynomials()
        self.left = ll(self.at)
        self.right = rl(self.at)

    def resetLines(self):
        self.left = np.zeros([1, 5])
        self.right = np.zeros([1, 5])

    def create(self):
        super(LineDetectorCamera, self).create()
        self.where = carla.Transform(carla.Location(x=2.5, z=1.3), carla.Rotation(pitch=-5.0))

    def invokeDraw(self):
        self.lineDetector().integrateLines()
        super(LineDetectorCamera, self).invokeDraw()

    def callBack(self, data):
        super().callBack(data)
        self.predict()
        self.lines()
        self.invokeDraw()


class SegmentationCamera(Camera):
    def __init__(self, manager, debug=False, show=True):
        super(SegmentationCamera, self).__init__(manager, debug, show)
        self.name = 'Semantic Segmentation'
        self.create()

    def create(self):
        super(SegmentationCamera, self).create()
        self.where = carla.Transform(carla.Location(x=2.5, z=0.7))

    def callBack(self, data):
        super(SegmentationCamera, self).callBack(data)
        self.invokeDraw()
