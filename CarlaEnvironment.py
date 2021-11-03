import glob
import html
import os
import sys
import random
import time
import numpy as np
import cv2

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

    #members
    client = None
    world = None
    blueprints = None
    model = None

    def __init__(self, camWidth, camHeight, debug=False):
        self.debug = debug
        self.camWidth = camWidth
        self.camHeight = camHeight
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(4.0)
        self.world = self.client.get_world()
        self.blueprints = self.world.get_blueprint_library()
        self.model = self.blueprints.filter('model3')[0]
        time.sleep(30)
        #self.run()

    def spawnVehicle(self):
        start = random.choice(self.world.get_map().get_spawn_points())
        vehicle = Vehicle(self, start)
        self.vehicles.append(vehicle)
        return vehicle

    def run(self):
        return None

    def __del__(self):
        for utility in self.utilities:
            try:
                utility.destroy()
                print("Removing utility!")
            except:
                print("Already deleted!")


class Vehicle:
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
        self.id = id
        self.environment = environment
        self.me = self.environment.world.spawn_actor(self.environment.model, spawnLocation)
        self.processMeasures()
        if self.debug:
            print("Vehicle {id} starting".format(self.id))

    def run(self):
        start = time.time()
        now = time.time()
        while now - start < MAX_TIME_CAR and not self.isColission:
            #there it will NN decide
            steer = random.uniform(-1, 1)
            throttle = random.uniform(0, 1)
            self.controlVehicle(throttle=throttle, steer=steer)
            self.processMeasures()
            if self.debug and self.frontView is not None:
                cv2.imshow("Camera", self.frontView)
                cv2.waitKey(1)

            time.sleep(0.05)
            now = time.time()

    def controlVehicle(self, throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False):
        if self.debug:
            print("{id}:[Control] T: {th}, S: {st}, B: {b}".format(id=self.id, th=throttle, st=steer, b=brake))
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake,
                                                        hand_brake=hand_brake, reverse=reverse))

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
        colsensor = self.blueprints.find('sensor.other.collision')
        where = carla.Transform(carla.Location(x=1.5, z=0.7))
        self.sCollision = self.environment.world.spawn_actor(colsensor, where, attach_to=self.me)
        self.sCollision.listen(lambda collision: self.processCollison(collision))
        self.actors.append(self.sCollision)

    def lidarSensor(self):
        lidar = self.blueprints.find('sensor.lidar.ray_cast')
        lidar.channels = 1
        where = carla.Transform(carla.Location(x=0, z=0))
        self.sLidar = self.environment.world.spawn_actor(lidar, where, attach_to=self.me)
        self.sLidar.listen(lambda lidarData: self.processLidarMeasure(lidarData))
        self.actors.append(self.sLidar)

    def processRGB(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.camHeight, self.camWidth, 4))
        im = i2[:, :, :3]
        #im2 = im.reshape((self.cam_height, self.cam_width))
        self.frontView = im

    def processCollison(self, collision):
        self.isColission = True

    def processLidarMeasure(self, lidarData):
        if self.debug:
            number = 0
            for location in lidarData:
                print("{num}: {location}".format(num=number, location=location))
                number += 1

    def print3Dvector(self, vector, type):
        x = vector.x
        y = vector.y
        z = vector.z
        print("{id}:[{t}}] X: {x}, Y: {y}, Z: {z}".format(id=self.id, t=type, x=x, y=y, z=z))

    def __del__(self):
        for actor in self.actors:
            actor.destroy()
            #self.actors.remove(actor)