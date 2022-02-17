import carla
import math
import cv2
import numpy as np
from carla import ColorConverter as cc
import queue
from PyQt5 import QtCore


class SensorManager(QtCore.QObject):
    """
    In case of Sync, we need to manage sensors when server ticks.
    If mode is async, we could simply rely on callbacks.
    """
    readySignal = QtCore.pyqtSignal()

    def __init__(self, vehicle, environment):
        super().__init__()
        self._queues = []
        self.sensors = []
        self.count = 0
        self.readySensors = 0
        self.vehicle = vehicle
        self.config = environment.config
        self.ldCam = LineDetectorCamera(self.vehicle, False)
        # self.seg = Camera(self, self.camHeight, self.camWidth, type='Semantic Segmentation')
        self.collision = CollisionSensor(self.vehicle, False)
        self.radar = RadarSensor(self.vehicle, False)
        self.lidar = LidarSensor(self.vehicle, False)
        self.obstacleDetector = ObstacleDetector(self.vehicle, False)
        #self.t = QtCore.QTimer()
        #self.t.setSingleShot(True)
        #self.t.timeout.connect(self.emitik)
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
            print(f"Activating {self.count}: {sensor.bp}")
            self.count += 1
            sensor.activate()
            sensor.ready.connect(self.readySensor)
        #self.t.start(1000)
        self.emitik()

    def emitik(self):
        #print("------------------------------------------------------------"+str(self.t.isActive()))
        for sensor in self.sensors:
            sensor.ready.emit()

    def on_world_tick(self):
        for sensor in self.sensors:
            sensor.on_world_tick()
        print("Skoncil som!")

    def readySensor(self):
        self.readySensors += 1
        print(f"Got signal ready from {self.readySensors}/{self.count}")
        self.checkForPossibleInvoke()

    def checkForPossibleInvoke(self):
        if self.readySensors == self.count:
            print("All sensors ready")
            self.readySensors = 0
            self.readySignal.emit()

    def isCollided(self):
        return self.collision.isCollided()

    def destroy(self):
        print(f"Invoking deletion of sensors of {self.vehicle.threadID} vehicle!")
        for sensor in self.sensors:
            print(f"Deleting sensor {sensor.bp}")
            sensor.destroy()


class Sensor(QtCore.QObject):
    debug: bool = False
    # send signal
    ready = QtCore.pyqtSignal()

    def __init__(self, vehicle, debug):
        super(Sensor, self).__init__()
        self.sensor = None
        self.bp = None
        self.where = None
        self.name = "Sensor"
        # self.ready = False
        self.queue = queue.Queue()
        self.vehicle = vehicle
        self.debug = debug

    def callBack(self):
        pass

    def activate(self):
        self.sensor = self.world().spawn_actor(self.bp, self.where, attach_to=self.reference())
        self.sensor.listen(lambda data: self.queue.put(data))

    def on_world_tick(self):
        print(f"[{self.name}] on world tick")
        if self.queue.qsize() == 0:
            print(f"Empty data queue for sensor {self.name}")
        else:
            self.callBack(self.queue.get())
        print(f"Emitting ready for sensor {self.name}")
        self.ready.emit()

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
        self.name = "Radar"
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
        self.name = "Collision"
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
        self.name = "Obstacle"
        self.bp = super().blueprints().find('sensor.other.obstacle')
        self.where = carla.Transform(carla.Location(x=1.5, z=0.7))

    def callBack(self, data):
        distance = data.distance
        print("{distance}".format(distance=distance))


class LidarSensor(Sensor):
    def __init__(self, vehicle, debug=False):
        super().__init__(vehicle, debug)
        self.name = "Lidar"
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
        # self.image.convert(self.options.get(self.name)[1])

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
