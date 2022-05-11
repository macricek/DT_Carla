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
    '''
    Handler of sensors. Holds reference to all sensors/cameras in list.
    Based on config, it runs the sensors objects.
    @author: Marko Chylik
    @Date: May, 2022
    '''

    def __init__(self, vehicle, environment):
        '''
        Create a SensorManager object.
        :param vehicle: reference to Vehicle's object
        :param environment: reference to CarlaEnvironment's object
        '''
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
        '''
        Based on config's settings add sensors to list.
        :return: None
        '''
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
        '''
        Activate all sensors from self.sensors list
        :return: None
        '''
        for sensor in self.sensors:
            self.print(f"Activating {self.count}: {sensor.name}")
            self.count += 1
            sensor.activate()

    def processSensors(self):
        '''
        When tick came from server, process all sensors.
        :return: None
        '''
        for sensor in self.sensors:
            sensor.on_world_tick()

    def isCollided(self):
        '''
        Fast reference to collision sensor's method
        :return: True if vehicle have collided, False if not
        '''
        return self.collision.isCollided()

    def lines(self) -> (np.ndarray, np.ndarray):
        '''
        Fast reference to LineDetection camera's left and right line
        :return: left [ndarray], right line [ndarray]
        '''
        retLeft = copy.deepcopy(self.ldCam.left)
        retRight = copy.deepcopy(self.ldCam.right)
        self.ldCam.resetLines()
        return retLeft, retRight

    def radarMeasurement(self) -> np.ndarray:
        '''
        Fast reference to Radar's averages ranges
        :return: average ranges of measures [ndarray]
        '''
        return self.radar.returnAverageRanges()

    def destroy(self):
        '''
        Destroy all current sensors in self.sensors
        :return: None
        '''
        self.print(f"Invoking deletion of sensors of {self.vehicle.vehicleID} vehicle!")
        for sensor in self.sensors:
            self.print(f"Deleting sensor {sensor.name}")
            sensor.destroy()

    def print(self, message):
        '''
        print message if Sensor Manager is in debug mode
        :param message: str to be printed
        :return: None
        '''
        if self.debug:
            print(message)

    def applyTesting(self):
        '''
        Apply test configuration:
        Show all attached cameras + print debug messages to console
        :return: None
        '''
        self.debug = True
        for camera in self.cameras:
            camera.show = True


class Sensor(QtCore.QObject):
    '''
    Main class for all Sensors -> all sensors needs to inherit this class for proper functionality of SensorManager!
    @author: Marko Chylik
    @Date: May, 2022
    '''
    debug: bool = False

    def __init__(self, manager, debug):
        '''
        Create sensor object
        :param manager: reference to SensorManager
        :param debug: bool
        '''
        super(Sensor, self).__init__()
        self.sensor = None
        self.bp = None
        self.where = None
        self.manager = manager
        self.name = "Sensor"
        self.queue = queue.Queue()
        self.vehicle = manager.vehicle
        self.debug = debug

    def callBack(self, data):
        '''
        Function that will handle data from sensor/camera later on
        :param data: sensor's data (blueprint)
        :return: None
        '''
        pass

    def activate(self):
        '''
        activate current sensor:
         - spawn it on world, attached to vehicle's BP
         - add any new incoming data to queue
        :return: None
        '''
        self.sensor = self.world().spawn_actor(self.bp, self.where, attach_to=self.reference())
        self.sensor.listen(lambda data: self.queue.put(data))

    def on_world_tick(self):
        '''
        call handler of data when any data are ready in queue
        :return: None
        '''
        if self.debug:
            print(f"[{self.name}] on world tick")
        if self.queue.qsize() > 0:
            self.callBack(self.queue.get())

    def reference(self):
        '''
        :return: Reference to vehicle's BP (real vehicle model in carla)
        '''
        return self.vehicle.ref()

    def setVehicle(self, vehicle):
        '''
        :param vehicle: Vehicle object
        :return: None
        '''
        self.vehicle = vehicle

    def blueprints(self):
        '''
        Fast reference to blueprints library
        :return: carla.Blueprints library
        '''
        return self.vehicle.environment.blueprints

    def world(self):
        '''
        Fast reference to world
        :return: carla.World
        '''
        return self.vehicle.environment.world

    def lineDetector(self):
        '''
        Fast reference to FALineDetector object
        :return: FALineDetector object
        '''
        return self.vehicle.fald

    def config(self):
        '''
        Fast reference to CarlaConfig object
        :return: CarlaConfig
        '''
        return self.vehicle.environment.config

    def destroy(self):
        '''
        Destroy self from world
        :return: None
        '''
        if self.sensor is not None:
            try:
                self.sensor.destroy()
            except:
                pass


class RadarSensor(Sensor):
    '''
    Radar Sensor
    @author: Marko Chylik
    @Date: May, 2022
    '''

    def __init__(self, manager, debug=False):
        '''
        Set basic attributes for Radar.
        Horizontal: (-45°; 45°)
        Vertical: (-12,5°; 12,5°)
        Max range: 50

        Create three ranges - Left, Center and Right

        :param manager: SensorManager
        :param debug: bool
        '''
        super().__init__(manager, debug)
        self.range = 50
        self.name = "Radar"
        self.bp = self.blueprints().find('sensor.other.radar')
        self.bp.set_attribute('horizontal_fov', str(90))
        self.bp.set_attribute('vertical_fov', str(25))
        self.bp.set_attribute('range', str(self.range))
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
        '''
        Handle radar data:
        Parse measures based on azimuth into left, right and center lists
        Calculate mean of these lists and store it in self.left, self.right and self.center

        :param data: carla.RadarDetection
        :return:
        '''
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
        '''
        :return: array consists of average distances in [left, center, right]
        '''
        return np.array([self.left, self.center, self.right])


class CollisionSensor(Sensor):
    '''
    Collision Sensor - detect collision in front of the vehicle
    @author: Marko Chylik
    @Date: May, 2022
    '''
    collided: bool

    def __init__(self, manager, debug=False):
        '''
        Set collided mode to false and init sensor
        :param manager: SensorManager
        :param debug: bool
        '''
        super().__init__(manager, debug)
        self.name = "Collision"
        self.collided = False
        self.bp = self.blueprints().find('sensor.other.collision')
        self.where = carla.Transform(carla.Location(x=1.5, z=0.7))

    def callBack(self, data):
        '''
        If there is incoming event, it means we have collided -> set collided to True
        :param data: carla.collisionEvent
        :return: None
        '''
        self.collided = True
        print("Vehicle {id} collided!".format(id=self.vehicle.vehicleID))

    def isCollided(self):
        '''
        :return: current collided state
        '''
        return self.collided


class ObstacleDetector(Sensor):
    '''
    Obstacle Detector - currently unused
    @author: Marko Chylik
    @Date: May, 2022
    '''
    def __init__(self, manager, debug=False):
        '''
        Init obstacle detector
        :param manager: SensorManager
        :param debug: bool
        '''
        super().__init__(manager, debug)
        self.name = "Obstacle"
        self.bp = self.blueprints().find('sensor.other.obstacle')
        self.where = carla.Transform(carla.Location(x=1.5, z=0.7))

    def callBack(self, data):
        '''
        parse distance to closest obstacle
        :param data: carla.obstacleMeasurement
        :return: None
        '''
        distance = data.distance
        print("{distance}".format(distance=distance))


class LidarSensor(Sensor):
    '''
    Lidar Sensor - currently unused
    @author: Marko Chylik
    @Date: May, 2022
    '''
    def __init__(self, manager, debug=False):
        '''
        Spawn lidar sensor into middle of vehicle with one channel
        :param manager: SensorManager
        :param debug: bool
        '''
        super().__init__(manager, debug)
        self.name = "Lidar"
        self.bp = self.blueprints().find('sensor.lidar.ray_cast')
        self.bp.channels = 1
        self.where = carla.Transform(carla.Location(x=0, y=0, z=0))

    def callBack(self, data):
        '''
        Parse lidar data. It's hard to filter, there are so many points...
        :param data: carla.LidarMeasurement
        :return: None
        '''
        if self.debug:
            number = 0
            for location in data:
                print("{num}: {location}".format(num=number, location=location))
                number += 1


class LaneInvasionDetector(Sensor):
    '''
    Lane Invasion Detector - used to know, when we cross line
    @author: Marko Chylik
    @Date: May, 2022
    '''

    def __init__(self, manager, debug=False):
        '''
        Init the LineInvasionDetector
        :param manager: SensorManager
        :param debug: bool
        set crossings counter to 0
        '''
        super().__init__(manager, debug)
        self.name = "LaneInvasion"
        self.bp = self.blueprints().find('sensor.other.lane_invasion')
        self.where = carla.Transform(carla.Location(x=0, y=0, z=0))

        self.crossings = 0
        self.lastCross = -5

    def callBack(self, data):
        '''
        add every crossing to crossings counter
        :param data: carla.LaneInvasionEvent
        :return:
        '''
        frameCrossed = data.frame
        if self.debug:
            print(f"Vehicle {self.vehicle.vehicleID} crossed line!")
            print(f"Crossed at frame {frameCrossed}, last cross was at {self.lastCross}")
        self.crossings += 1
        self.lastCross = frameCrossed


class Camera(Sensor):
    '''
    Basic Camera - default usage as watcher from behind of the car
    @author: Marko Chylik
    @Date: May, 2022

    Define options dictionary to apply default settings for all possible cameras.
    '''

    name: str

    options = {
        'Main': ['sensor.camera.rgb', cc.Raw],
        'Depth': ['sensor.camera.depth', cc.LogarithmicDepth],
        'Semantic Segmentation': ['sensor.camera.semantic_segmentation', cc.CityScapesPalette],
        'LineDetection': ['sensor.camera.rgb', cc.Raw]
    }

    def __init__(self, manager, debug=False, show=True):
        '''
        Init Camera.
        :param manager: SensorManager
        :param debug: bool
        :param show: bool -> if true, camera's image will be displayed

        Set height and width of image based on config
        Set bounding box around the car to calculate proper middle of the 3D area
        '''

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
        '''
        Create camera based on name of the camera, then apply default settings from options dictionary
        :return: None
        '''
        typeOfCamera = self.options.get(self.name)[0]
        self.bp = self.blueprints().find(typeOfCamera)
        self.bp.set_attribute('image_size_x', f'{self.camWidth}')
        self.bp.set_attribute('image_size_y', f'{self.camHeight}')
        self.bp.set_attribute('fov', '110')
        self.where = carla.Transform(carla.Location(x=-2.0 * self.bound_x, y=+0.0 * self.bound_y, z=2.0 * self.bound_z),
                                     carla.Rotation(pitch=-8.0))

    def callBack(self, data):
        '''
        handle Camera's output - convert image data, if we are using some special camera
        :param data: carla.Image
        :return:
        '''
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
        !IN SEPARATE THREAD!
        Show the image from camera using openCV imshow.
        When stop is coming from SensorManager, we need to break the infinite while
        :return: None
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
        '''
        Invoke draw - start separate thread. Ensure, that could happen just once per every camera!
        :return: None
        '''
        if self.show and self.drawingThread is None:
            self.drawingThread = threading.Thread(target=self.draw)
            self.drawingThread.start()

    def isMain(self):
        '''
        This is determined by camera's name
        :return: bool
        '''
        return self.name == 'Main'

    def destroy(self):
        '''
        destroy Camera, but first wait until thread that is drawing the camera's image is finished
        :return:
        '''
        if self.drawingThread is not None:
            self.show = False
            self.stop = True  # break showing threads
            for _ in range(10):
                if self.stop:
                    self.manager.environment.tick()
            self.drawingThread = None
        super(Camera, self).destroy()


class LineDetectorCamera(Camera):
    '''
    Special camera in front of vehicle used to line detection
    @author: Marko Chylik
    @Date: May, 2022
    '''
    left: np.ndarray
    right: np.ndarray

    def __init__(self, manager, debug=False, show=True):
        '''
        Init of this camera. Use Camera's attributes
        :param manager: SensorManager
        :param debug: bool
        :param show: bool

        set name to LineDetection
        set self.at to [0, 7.5, 15]. It means that polynomial approximation will be used with these as x.
        y = a3 * x^3 + a2 * x^2 + a1 * x + a0
        '''
        super(LineDetectorCamera, self).__init__(manager, debug, show)
        self.name = 'LineDetection'
        self.create()
        self.at = np.array([0, 7.5, 15])
        self.resetLines()

    def predict(self):
        '''
        loads image to LineDetector and predicts where are lines
        :return: None
        '''
        self.lineDetector().loadImage(numpyArr=self.image)
        self.lineDetector().predict()

    def lines(self):
        '''
        Left and right points in range "self.at"
        :return: None
        '''
        ll, rl = self.lineDetector().extractPolynomials()
        self.left = self.validateLine(ll(self.at))
        self.right = self.validateLine(rl(self.at))

    @staticmethod
    def validateLine(line: np.ndarray) -> np.ndarray:
        '''
        Validate, if LineDetection is accurate based on statistical methods:
        if maximum value of line is more than 3, Lines are invalid.
        if variance of lines are more than 2, Lines are invalid.

        :param line: ndarray of 3 points based on polynomials.
        :return: line, if line is valid. If not, return zeros.
        '''
        variance = np.var(line)
        maxVal = np.max(np.abs(line))

        if maxVal > 3:
            return np.zeros(line.shape)
        if variance > 2:
            return np.zeros(line.shape)

        return line

    def resetLines(self):
        '''
        After lines are read, reset them to zeros, to not use them more than once
        :return: None
        '''
        self.left = np.zeros([1, len(self.at)])
        self.right = np.zeros([1, len(self.at)])

    def create(self):
        '''
        Create Camera for line detection purposes
        :return:
        '''
        super(LineDetectorCamera, self).create()
        self.where = carla.Transform(carla.Location(x=2.5, z=1.3), carla.Rotation(pitch=-5.0))

    def invokeDraw(self):
        '''
        Integrate lines towards the image that will be shown.
        :return:
        '''
        self.lineDetector().integrateLines()
        super(LineDetectorCamera, self).invokeDraw()

    def callBack(self, data):
        '''
        Handle incoming image data:
            1) predict the lines on u x v image
            2) get lines points based on polynomial approximation
            3) draw them towards the data (to show them properly)

        :param data: carla.Image
        :return: None
        '''
        super().callBack(data)
        self.predict()
        self.lines()
        self.invokeDraw()


class SegmentationCamera(Camera):
    '''
    Segmentation Camera - not used
    @author: Marko Chylik
    @Date: May, 2022
    '''
    def __init__(self, manager, debug=False, show=True):
        '''
        Init camera -> set name to Semantic Segmentation
        :param manager: SensorManager
        :param debug: bool
        :param show: bool
        '''
        super(SegmentationCamera, self).__init__(manager, debug, show)
        self.name = 'Semantic Segmentation'
        self.create()

    def create(self):
        '''
        Create camera with specific image options from directory
        :return:
        '''
        super(SegmentationCamera, self).create()
        self.where = carla.Transform(carla.Location(x=2.5, z=0.7))

    def callBack(self, data):
        '''
        Handle data -> just draw it.
        :param data: carla.Image
        :return:
        '''
        super(SegmentationCamera, self).callBack(data)
        self.invokeDraw()
