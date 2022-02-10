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
    """

    def __init__(self, vehicle, environment):
        self._queues = []
        self.sensors = []
        self.vehicle = vehicle
        self.config = environment.config
        self.ldCam = LineDetectorCamera(self.vehicle)
        #self.seg = Camera(self, self.camHeight, self.camWidth, type='Semantic Segmentation')
        self.collision = CollisionSensor(self.vehicle)
        self.radar = RadarSensor(self.vehicle)
        self.lidar = LidarSensor(self.vehicle)
        self.obstacleDetector = ObstacleDetector(self.vehicle)

        self.addToSensorsList()
        self.activate()

    def addToSensorsList(self):
        settings: dict
        settings = self.config.readSection("Sensors")
        if bool(settings.get("radarsensor")):
            self.sensors.append(self.radar)
        if bool(settings.get("linedetectorcamera")):
            self.sensors.append(self.ldCam)
        if bool(settings.get("collisionsensor")):
            self.sensors.append(self.collision)
        if bool(settings.get("obstacledetector")):
            self.sensors.append(self.obstacleDetector)
        if bool(settings.get("lidarsensor")):
            self.sensors.append(self.lidar)

    def activate(self):
        for sensor in self.sensors:
            sensor.activate()

    def on_world_tick(self):
        for sensor in self.sensors:
            sensor.on_world_tick()

    def isCollided(self):
        return self.collision.isCollided()


class Sensor(object):
    debug: bool = False

    def __init__(self, vehicle, debug):
        self.sensor = None
        self.bp = None
        self.where = None
        self.ready = False
        self.vehicle = vehicle
        self.debug = debug

    def activate(self):
        self.sensor = self.world().spawn_actor(self.bp, self.where, attach_to=self.reference())
        self.sensor.listen(lambda data: self.callBack(data))

    def on_world_tick(self):
        self.ready = False

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
        self.camHeight = int(d["height"])
        self.camWidth = int(d["width"])
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
        #self.image.convert(self.options.get(self.name)[1])

    def draw(self):
        cv2.imshow("Vehicle {id}, Camera {n}".format(id=self.vehicle.threadID, n=self.name), self.image)
        cv2.waitKey(1)


class LineDetectorCamera(Camera):
    def __init__(self, vehicle, debug=False):
        super(LineDetectorCamera, self).__init__(vehicle, debug)
        self.name = 'LineDetection'
        self.create()

    def predict(self):
        self.lineDetector().loadImage(numpyArr=self.image)
        self.lineDetector().predict()
        self.lineDetector().integrateLines()

    def callBack(self, data):
        super().callBack(data)
        self.predict()
        self.draw()
