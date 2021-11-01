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

actor_list = []


def printLocation(location):
    x = location.x
    y = location.y
    z = location.z
    print("X: {x}, Y: {y}, Z: {z}".format(x=x, y=y, z=z))


class CarlaCar:
    SHOW_CAM = False

    debug = True

    # lists
    utilities = []  # list of "actors"
    collisions = []

    # other members
    client = None
    world = None
    blueprints = None
    model = None
    vehicle = None

    # sensors with callbacks
    sCam = None
    sCollision = None
    sLidar = None

    # positions
    start = None
    location = None

    # camera images
    frontView = None

    def __init__(self, IM_WIDTH, IM_HEIGHT, debug=False):
        self.debug = debug
        self.cam_width = IM_WIDTH
        self.cam_height = IM_HEIGHT
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(4.0)
        self.world = self.client.get_world()
        self.blueprints = self.world.get_blueprint_library()
        self.model = self.blueprints.filter('model3')[0]
        self.spawn()
        self.location = self.vehicle.get_location()
        self.rgbCameraSensor()
        self.collisionSensor()
        #self.lidarSensor()
        self.controlVehicle(1, 0)
        self.run()

    def spawn(self):
        self.start = random.choice(self.world.get_map().get_spawn_points())
        if self.debug:
            print("Spawn location")
            printLocation(self.start.location)
        self.vehicle = self.world.spawn_actor(self.model, self.start)
        self.utilities.append(self.vehicle)

    def rgbCameraSensor(self):
        camera = self.blueprints.find('sensor.camera.rgb')
        camera.set_attribute('image_size_x', f'{self.cam_width}')
        camera.set_attribute('image_size_y', f'{self.cam_height}')
        camera.set_attribute('fov', '110')
        where = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sCam = self.world.spawn_actor(camera, where, attach_to=self.vehicle)
        self.utilities.append(self.sCam)
        self.sCam.listen(lambda data: self.processRGB(data))

    def collisionSensor(self):
        colsensor = self.blueprints.find('sensor.other.collision')
        where = carla.Transform(carla.Location(x=1.5, z=0.7))
        self.sCollision = self.world.spawn_actor(colsensor, where, attach_to=self.vehicle)
        self.sCollision.listen(lambda collision: self.processCollison(collision))
        self.utilities.append(self.sCollision)

    def lidarSensor(self):
        lidar = self.blueprints.find('sensor.lidar.ray_cast')
        lidar.channels = 1
        where = carla.Transform(carla.Location(x=0, z=0))
        self.sLidar = self.world.spawn_actor(lidar, where, attach_to=self.vehicle)
        self.sLidar.listen(lambda lidarData: self.processLidarMeasure(lidarData))
        self.utilities.append(self.sLidar)

    def controlVehicle(self, throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False):
        if self.debug:
            print("[Control Vehicle] T: {th}, S: {st}, B: {b}, HB: {hb}, R: {r}".format(th=throttle, st=steer, b=brake,
                                                                                        hb=hand_brake, r=reverse))
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake,
                                                        hand_brake=hand_brake, reverse=reverse))

    def processRGB(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.cam_height, self.cam_width, 4))
        im = i2[:, :, :3]
        #im2 = im.reshape((self.cam_height, self.cam_width))
        if self.SHOW_CAM:
            cv2.imshow("Camera", im)
            cv2.waitKey(1)
        self.frontView = im

    def processCollison(self, collision):
        self.collisions.append(collision)

    def run(self):
        start = time.time()
        now = time.time()
        while now - start < 15 and not self.collisions:
            self.location = self.vehicle.get_location()
            printLocation(self.location)
            if self.frontView is not None:
                cv2.imshow("Camera", self.frontView)
                cv2.waitKey(1)
            time.sleep(0.05)
            now = time.time()

    def processLidarMeasure(self, lidarData):
        if self.debug:
            number = 0
            for location in lidarData:
                print("{num}: {location}".format(num=number, location=location))
                number += 1

    def __del__(self):
        for utility in self.utilities:
            try:
                utility.destroy()
                print("Removing utility!")
            except:
                print("Already deleted!")
