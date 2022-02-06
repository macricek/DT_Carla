import enum
import carla
import math
import cv2
import numpy as np
from carla import ColorConverter as cc
import queue


class SensorManager(object):
    """
    In case of Sync, we need to manage sensors when server ticks.
    If mode is async, we could simply rely on callbacks.
     def setupSensors(self):
        self.environment.allFeatures.append(self.me)
        self.ldCam = LineDetectorCamera(self)
        # self.seg = Camera(self, self.camHeight, self.camWidth, type='Semantic Segmentation')
        self.collision = CollisionSensor(self)
        # self.radar = RadarSensor(self, True)
        # self.lidar = LidarSensor(self)
        self.obstacleDetector = ObstacleDetector(self)
        self.sensors.append(self.ldCam)
        # self.sensors.append(self.seg)
        self.sensors.append(self.collision)
        # self.sensors.append(self.radar)
        # self.sensors.append(self.lidar)
        self.sensors.append(self.obstacleDetector)
        self.environment.allFeatures.extend(self.sensors)
    """


    def __init__(self, *sensors):
        self._queues = []


class Sensor(object):
    debug: bool = False

    def __init__(self, vehicle, debug):
        self.sensor = None
        self.bp = None
        self.where = None
        self.vehicle = vehicle
        self.debug = debug

    def activate(self):
        self.sensor = self.world().spawn_actor(self.bp, self.where, attach_to=self.reference())
        self.sensor.listen(lambda data: self.callBack(data))

    def callBack(self, data):
        print("Default callback!")

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
    def __init__(self, vehicle, debug=False):
        super().__init__(vehicle, debug)
        self.velocity_range = 7.5
        self.bp = self.blueprints().find('sensor.other.radar')
        self.bp.set_attribute('horizontal_fov', str(35))
        self.bp.set_attribute('vertical_fov', str(20))
        self.where = carla.Transform(carla.Location(x=1.5))

    def callBack(self, data):
        current_rot = data.transform.rotation
        i = 0
        for detect in data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            dist = detect.depth
            if self.debug:
                print("Dist to {i}: {d}, Azimuth: {a}, Altitude: {al}".format(i=i, d=dist, a=azi, al=alt))
                i += 1


class CollisionSensor(Sensor):
    collided: bool

    def __init__(self, vehicle, debug=False):
        super().__init__(vehicle, debug)
        self.collided = False
        self.bp = super().blueprints().find('sensor.other.collision')
        self.where = carla.Transform(carla.Location(x=1.5, z=0.7))

    def callBack(self, data):
        self.collided = True
        print("Vehicle {id} collided!".format(id=self.vehicle.threadID))

    def isCollided(self):
        return self.collided


class ObstacleDetector(Sensor):
    def __init__(self, vehicle, debug=False):
        super().__init__(vehicle, debug)
        self.bp = super().blueprints().find('sensor.other.obstacle')
        self.where = carla.Transform(carla.Location(x=1.5, z=0.7))

    def callBack(self, data):
        distance = data.distance
        print("{distance}".format(distance=distance))


class LidarSensor(Sensor):
    def __init__(self, vehicle, debug=False):
        super().__init__(vehicle, debug)
        self.bp = super().blueprints().find('sensor.lidar.ray_cast')
        self.bp.channels = 1
        self.where = carla.Transform(carla.Location(x=0, z=0))

    def callBack(self, data):
        if self.debug:
            number = 0
            for location in data:
                print("{num}: {location}".format(num=number, location=location))
                number += 1


class Camera(Sensor):
    name: str

    options = {
        'RGB': ['sensor.camera.rgb', cc.Raw],
        'Depth': ['sensor.camera.depth', cc.LogarithmicDepth],
        'Semantic Segmentation': ['sensor.camera.semantic_segmentation', cc.CityScapesPalette],
        'LineDetection': ['sensor.camera.rgb', cc.Raw]
    }

    def __init__(self, vehicle, debug=False):
        super().__init__(vehicle, debug)
        d = self.config().readSection('Camera')
        self.camHeight = d["height"]
        self.camWidth = d["width"]
        self.image = None
        self.name = 'RGB'

    def create(self):
        typeOfCamera = self.options.get(self.name)[0]
        self.bp = super().blueprints().find(typeOfCamera)
        self.bp.set_attribute('image_size_x', f'{self.camWidth}')
        self.bp.set_attribute('image_size_y', f'{self.camHeight}')
        self.bp.set_attribute('fov', '110')
        self.where = carla.Transform(carla.Location(x=2.5, z=0.7))

    def callBack(self, data):
        i = np.array(data.raw_data)
        i2 = i.reshape((self.camHeight, self.camWidth, 4))
        self.image = i2[:, :, :3]
        self.image.convert(self.options.get(self.name)[1])

    def draw(self):
        cv2.imshow("Vehicle {id}, Camera {n}".format(id=self.vehicle.threadID, n=self.name), self.image)
        cv2.waitKey(1)


class LineDetectorCamera(Camera):
    def __init__(self, vehicle, debug=False):
        super(LineDetectorCamera, self).__init__(vehicle, debug)
        self.key = 'LineDetection'

    def predict(self):
        self.lineDetector().loadImage(numpyArr=self.image)
        self.lineDetector().predict()
        self.lineDetector().integrateLines()

    def callBack(self, data):
        data.convert(self.options.get())
        super().callBack(data)
        self.predict()
        self.draw()
