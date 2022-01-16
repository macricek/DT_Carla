import glob
import threading
import os
import sys
import random
import time
import numpy as np
import cv2
import math
from carla import ColorConverter as cc

import LineDetection
from LineDetection import CNNLineDetector, transformImage

# from Carla doc
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

## global constants
MAX_TIME_CAR = 30


class CarlaEnvironment:
    debug = True
    editEnvironment = False
    # lists
    vehicles = []
    allFeatures = []
    maxId = 50
    #members
    client = None
    world = None
    blueprints = None
    model = None
    cnnLineDetector = None

    def __init__(self, numVehicles, debug=False):
        self.id = 0
        self.debug = debug
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(4.0)
        self.world = self.client.get_world()
        self.cnnLineDetector = CNNLineDetector(from_scratch=False, dataPath=LineDetection.data_path)
        if self.editEnvironment:
            self.setSimulation()

        self.blueprints = self.world.get_blueprint_library()
        self.model = self.blueprints.filter('model3')[0]
        self.map = self.world.get_map()
        self.spawnVehicles(numVehicles)
        self.startThreads()

    def spawnVehicles(self, numVehicles):
        for i in range(0, numVehicles):
            spawnPoints = self.map.get_spawn_points()
            start = spawnPoints[self.id]
            vehicle = Vehicle(self, start, id=self.id)
            self.vehicles.append(vehicle)
            self.allFeatures.append(vehicle)
            self.id += 1
            time.sleep(1)

    def setSimulation(self):
        self.world.unload_map_layer(carla.MapLayer.All) #make map easier
        self.world.load_map_layer(carla.MapLayer.Walls)
        self.originalSettings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

    def stopSimulation(self):
        self.world.load_map_layer(carla.MapLayer.All)
        self.world.apply_settings(self.originalSettings)

    def startThreads(self):
        for vehicle in self.vehicles:
            vehicle.start()

    def run(self):
        for vehicle in self.vehicles:
            vehicle.run()

    def deleteVehicle(self, vehicle):
        for v in self.vehicles:
            if v == vehicle:
                try:
                    self.vehicles.remove(vehicle)
                    del vehicle
                except:
                    print("already out")
                finally:
                    break

    def deleteAll(self):
        print("Destroying")
        if self.editEnvironment:
            self.stopSimulation()
        for actor in self.allFeatures:
            try:
                actor.destroy()
                print("Removing utility!")
            except:
                print("Already deleted!")


class Vehicle(threading.Thread):
    me = None #ref to vehicle
    environment = None #ref to environment upper

    camHeight = 512
    camWidth = 1024

    #states of vehicle
    location = None
    velocity = None

    #camera
    __rgb = None
    __seg = None

    #sensor
    __collision = None
    __lidar = None
    __radar = None
    __obstacleDetector = None

    sensors = []

    def __init__(self, environment, spawnLocation, id):
        threading.Thread.__init__(self)
        self.threadID = id  # threadOBJ
        self.environment = environment
        self.debug = self.environment.debug
        self.me = self.environment.world.spawn_actor(self.environment.model, spawnLocation)
        self.setupSensors()
        self.processMeasures()
        if self.debug:
            print("Vehicle {id} starting".format(id=self.threadID))

    def run(self):
        start = time.time()
        now = time.time()
        while now - start < MAX_TIME_CAR and not self.isCollided() and self.me:
            #there it will NN decide
            steer = random.uniform(-1, 1)
            throttle = random.uniform(0, 1)
            try:
                self.controlVehicle(throttle=throttle, steer=steer)
                self.processMeasures()
                if self.debug and self.__rgb.isImageAvailable():
                    self.__rgb.draw()
                if self.debug and self.__seg.isImageAvailable():
                    self.__seg.draw()
                time.sleep(0.05)
                now = time.time()
            except:
                None
        self.destroy()

    def controlVehicle(self, throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False):
        if self.debug:
            print("{id}:[Control] T: {th}, S: {st}, B: {b}".format(id=self.threadID, th=throttle, st=steer, b=brake))
        self.me.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake,
                                                        hand_brake=hand_brake, reverse=reverse))

    def isCollided(self):
        return self.__collision.isCollided()

    def setupSensors(self):
        self.environment.allFeatures.append(self.me)
        self.__rgb = Camera(self, self.camHeight, self.camWidth)
        #self.__seg = Camera(self, self.camHeight, self.camWidth, type='Semantic Segmentation')
        self.__collision = CollisionSensor(self)
        #self.__radar = RadarSensor(self, True)
        #self.__lidar = LidarSensor(self)
        self.__obstacleDetector = ObstacleDetector(self)
        self.sensors.append(self.__rgb)
        #self.sensors.append(self.__seg)
        self.sensors.append(self.__collision)
        #self.sensors.append(self.__radar)
        #self.sensors.append(self.__lidar)
        self.sensors.append(self.__obstacleDetector)
        self.environment.allFeatures.extend(self.sensors)

    def processMeasures(self):
        self.location = self.me.get_location()
        self.velocity = self.me.get_velocity()
        if self.debug:
            self.print3Dvector(self.location, "Location")
            self.print3Dvector(self.velocity, "Velocity")

    def ref(self):
        return self.me

    def print3Dvector(self, vector, type):
        x = vector.x
        y = vector.y
        z = vector.z
        print("{id}:[{t}] X: {x}, Y: {y}, Z: {z}".format(id=self.threadID, t=type, x=x, y=y, z=z))

    def destroy(self):
        if self.debug:
            print("Destroying Vehicle {id}".format(id=self.threadID))
        try:
            for actor in self.sensors:
                actor.destroy()
            self.me.destroy()
        finally:
            self.environment.deleteVehicle(self)


class Sensor(object):
    sensor = None
    vehicle: Vehicle
    debug: bool = False

    def __init__(self, vehicle, debug):
        self.sensor = None
        self.vehicle = vehicle
        self.debug = debug

    def setSensor(self, sensor):
        self.sensor = sensor

    def reference(self):
        return self.vehicle.ref()

    def setVehicle(self, vehicle):
        self.vehicle = vehicle

    def blueprints(self):
        return self.vehicle.environment.blueprints

    def world(self):
        return self.vehicle.environment.world

    def lineDetector(self):
        return self.vehicle.environment.cnnLineDetector

    def destroy(self):
        if self.sensor is not None:
            try:
                self.sensor.destroy()
            except:
                None


class RadarSensor(Sensor):
    def __init__(self, vehicle, debug=False):
        super().__init__(vehicle, debug)
        self.velocity_range = 7.5
        bp = self.blueprints().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35))
        bp.set_attribute('vertical_fov', str(20))
        where = carla.Transform(carla.Location(x=1.5))
        self.setSensor(self.world().spawn_actor(bp, where, attach_to=self.reference()))
        self.sensor.listen(
            lambda radar_data: self._Radar_callback(radar_data))

    def _Radar_callback(self, radar_data):
        current_rot = radar_data.transform.rotation
        i = 0
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            dist = detect.depth
            if self.debug:
                print("Dist to {i}: {d}, Azimuth: {a}, Altitude: {al}".format(i=i, d=dist, a=azi, al=alt))
                i+=1


class Camera(Sensor):
    _camHeight = None
    _camWidth = None
    image = None
    detectedLines = None

    options = {
        'RGB': ['sensor.camera.rgb', cc.Raw],
        'Depth': ['sensor.camera.depth', cc.LogarithmicDepth],
        'Semantic Segmentation': ['sensor.camera.semantic_segmentation', cc.CityScapesPalette]
    }

    def __init__(self, vehicle, height, width, type='RGB', debug=False):
        self._camHeight = height
        self._camWidth = width
        super().__init__(vehicle, debug)
        self.option = self.options.get(type)
        self.type = type
        camera = super().blueprints().find(self.option[0])
        camera.set_attribute('image_size_x', f'{self._camWidth}')
        camera.set_attribute('image_size_y', f'{self._camHeight}')
        camera.set_attribute('fov', '110')

        where = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.setSensor(self.world().spawn_actor(camera, where, attach_to=self.reference()))
        self.sensor.listen(lambda image: self._cameraCallback(image))

    def _cameraCallback(self, image):
        image.convert(self.option[1])
        i = np.array(image.raw_data)
        i2 = i.reshape((self._camHeight, self._camWidth, 4))
        self.image = i2[:, :, :3]
        #if self.type.startswith('R'):
            #self.predict()

    def predict(self):
        shapeIm = np.shape(self.image)
        torchImage, _ = transformImage(image=self.image, transformation=LineDetection.testtransform, mask=np.empty(shapeIm))
        nnMask = self.lineDetector().predict(torchImage)
        detectedLines = nnMask.cpu().numpy().transpose(1, 2, 0)
        return detectedLines

    def isImageAvailable(self):
        return self.image is not None

    def draw(self):
        cv2.imshow("Vehicle {id}, Camera {n}".format(id=self.vehicle.threadID, n=self.type), self.image)
        #cv2.imshow("Lines", self.detectedLines)
        #self.predict()
        cv2.waitKey(1)


class CollisionSensor(Sensor):
    __collided: bool

    def __init__(self, vehicle, debug=False):
        super().__init__(vehicle, debug)
        self.__collided = False
        colsensor = super().blueprints().find('sensor.other.collision')
        where = carla.Transform(carla.Location(x=1.5, z=0.7))
        self.setSensor(self.world().spawn_actor(colsensor, where, attach_to=self.reference()))
        self.sensor.listen(lambda collision: self.processCollison(collision))

    def processCollison(self, collision):
        self.__collided = True
        print("Vehicle {id} collided!".format(id=self.vehicle.threadID))

    def isCollided(self):
        return self.__collided


class ObstacleDetector(Sensor):
    def __init__(self, vehicle, debug=False):
        super().__init__(vehicle, debug)
        obsDetector = super().blueprints().find('sensor.other.obstacle')
        where = carla.Transform(carla.Location(x=1.5, z=0.7))
        self.setSensor(self.world().spawn_actor(obsDetector, where, attach_to=self.reference()))
        self.sensor.listen(lambda obstacle: self.processObstacle(obstacle))

    @staticmethod
    def processObstacle(obstacle):
        distance = obstacle.distance
        print("{distance}".format(distance=distance))


class LidarSensor(Sensor):
    def __init__(self, vehicle, debug=False):
        super().__init__(vehicle, debug)
        lidar = super().blueprints().find('sensor.lidar.ray_cast')
        lidar.channels = 1
        where = carla.Transform(carla.Location(x=0, z=0))
        self.setSensor(self.world().spawn_actor(lidar, where, attach_to=self.reference()))
        self.sensor.listen(lambda lidarData: self.processLidarMeasure(lidarData))

    def processLidarMeasure(self, lidarData):
        if self.debug:
            number = 0
            for location in lidarData:
                print("{num}: {location}".format(num=number, location=location))
                number += 1