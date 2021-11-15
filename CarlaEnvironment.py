import glob
import threading
import os
import sys
import random
import time
import numpy as np
import cv2
import math
import weakref

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

    # lists
    vehicles = []
    allFeatures = []
    maxId = 50
    #members
    client = None
    world = None
    blueprints = None
    model = None

    def __init__(self, camWidth, camHeight, numVehicles, debug=False):
        self.id = 0
        self.debug = debug
        self.camWidth = camWidth
        self.camHeight = camHeight
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(4.0)
        self.world = self.client.get_world()
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

    #states of vehicle
    location = None
    velocity = None

    #camera
    sCam = None
    camWidth = None
    camHeight = None
    frontView = None

    #collision sensor
    sCollision = None
    isColission = False

    #lidarsensor
    sLidar = None

    actors = []

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
        while now - start < MAX_TIME_CAR and not self.isColission and self.me:
            #there it will NN decide
            steer = random.uniform(-1, 1)
            throttle = random.uniform(0, 1)
            self.controlVehicle(throttle=throttle, steer=steer)
            self.processMeasures()
            if self.debug and self.frontView is not None:
                cv2.imshow("Vehicle {id}".format(id=self.threadID), self.frontView)
                cv2.waitKey(1)

            time.sleep(0.05)
            now = time.time()
        self.destroy()

    def controlVehicle(self, throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False):
        if self.debug:
            print("{id}:[Control] T: {th}, S: {st}, B: {b}".format(id=self.threadID, th=throttle, st=steer, b=brake))
        self.me.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake,
                                                        hand_brake=hand_brake, reverse=reverse))

    def setupSensors(self):
        self.rgbCameraSensor()
        self.collisionSensor()
        self.testSensor = RadarSensor(self)
        self.actors.append(self.testSensor)
        self.environment.allFeatures.append(self.me)
        self.environment.allFeatures.extend(self.actors)

    def processMeasures(self):
        self.location = self.me.get_location()
        self.velocity = self.me.get_velocity()
        if self.debug:
            self.print3Dvector(self.location, "Location")
            self.print3Dvector(self.velocity, "Velocity")

    def rgbCameraSensor(self):
        self.camHeight = self.environment.camHeight
        self.camWidth = self.environment.camWidth
        camera = self.environment.blueprints.find('sensor.camera.rgb')
        camera.set_attribute('image_size_x', f'{self.camWidth}')
        camera.set_attribute('image_size_y', f'{self.camHeight}')
        camera.set_attribute('fov', '110')
        where = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sCam = self.environment.world.spawn_actor(camera, where, attach_to=self.me)
        self.actors.append(self.sCam)
        self.sCam.listen(lambda data: self.processRGB(data))

    def collisionSensor(self):
        colsensor = self.environment.blueprints.find('sensor.other.collision')
        where = carla.Transform(carla.Location(x=1.5, z=0.7))
        self.sCollision = self.environment.world.spawn_actor(colsensor, where, attach_to=self.me)
        self.sCollision.listen(lambda collision: self.processCollison(collision))
        self.actors.append(self.sCollision)

    def lidarSensor(self):
        lidar = self.environment.blueprints.find('sensor.lidar.ray_cast')
        lidar.channels = 1
        where = carla.Transform(carla.Location(x=0, z=0))
        self.sLidar = self.environment.world.spawn_actor(lidar, where, attach_to=self.me)
        self.sLidar.listen(lambda lidarData: self.processLidarMeasure(lidarData))
        self.actors.append(self.sLidar)

    def radarSensor(self):
        # blueprint attribute	Type	Default	Description
        # horizontal_fov	float	30.0	Horizontal field of view in degrees.
        # points_per_second	int	1500	Points generated by all lasers per second.
        # range	float	        100	Maximum distance to measure/raycast in meters.
        # sensor_tick	float	0.0	Simulation seconds between sensor captures (ticks).
        # vertical_fov	float	30.0	Vertical field of view in degrees.
        radar = self.environment.blueprints.find('sensor.other.radar')
        radar.range = 10
        radar.vertical_fov = 60
        where = carla.Transform(carla.Location(x=1.5))
        self.sRadar = self.environment.world.spawn_actor(radar, where, attach_to=self.me)
        self.sRadar.listen(lambda radarData: self.processRadarData(radarData))
        self.actors.append(self.sRadar)

    def processRGB(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.camHeight, self.camWidth, 4))
        im = i2[:, :, :3]
        #im2 = im.reshape((self.cam_height, self.cam_width))
        self.frontView = im

    def processCollison(self, collision):
        self.isColission = True
        if self.debug:
            print("Vehicle {id} collided!".format(id=self.threadID))

    def processLidarMeasure(self, lidarData):
        if self.debug:
            number = 0
            for location in lidarData:
                print("{num}: {location}".format(num=number, location=location))
                number += 1

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
            for actor in self.actors:
                actor.destroy()
            self.me.destroy()
        finally:
            self.environment.deleteVehicle(self)


class Sensor(object):
    sensor = None

    def __init__(self):
        self.sensor = None

    def setSensor(self, sensor):
        self.sensor = sensor

    def destroy(self):
        if self.sensor is not None:
            self.sensor.destroy()


class RadarSensor(Sensor):
    def __init__(self, parent_actor):
        self.parent = parent_actor

        self.velocity_range = 7.5 # m/s
        world = self.parent.environment.world
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35))
        bp.set_attribute('vertical_fov', str(20))
        where = carla.Transform(carla.Location(x=1.5))
        super().setSensor(world.spawn_actor(bp, where, attach_to=self.parent.ref()))
        # We need a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        # points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        # points = np.reshape(points, (len(radar_data), 4))

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)