import carla
#from Vehicle import Vehicle
import math
import cv2
import numpy as np
from carla import ColorConverter as cc


class Sensor(object):
    sensor = None
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
        return self.vehicle.fald

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
    image = None
    detectedLines = None

    options = {
        'RGB': ['sensor.camera.rgb', cc.Raw],
        'Depth': ['sensor.camera.depth', cc.LogarithmicDepth],
        'Semantic Segmentation': ['sensor.camera.semantic_segmentation', cc.CityScapesPalette]
    }

    def __init__(self, vehicle, height, width, type='RGB', debug=False):
        self.camHeight = height
        self.camWidth = width
        super().__init__(vehicle, debug)
        self.option = self.options.get(type)
        self.type = type
        camera = super().blueprints().find(self.option[0])
        camera.set_attribute('image_size_x', f'{self.camWidth}')
        camera.set_attribute('image_size_y', f'{self.camHeight}')
        camera.set_attribute('fov', '110')

        where = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.setSensor(self.world().spawn_actor(camera, where, attach_to=self.reference()))
        self.sensor.listen(lambda image: self.cameraCallback(image))

    def cameraCallback(self, image):
        image.convert(self.option[1])
        i = np.array(image.raw_data)
        i2 = i.reshape((self.camHeight, self.camWidth, 4))
        self.image = i2[:, :, :3]
        self.lineDetector().pred

    def isImageAvailable(self):
        return self.image is not None

    def draw(self):
        cv2.imshow("Vehicle {id}, Camera {n}".format(id=self.vehicle.threadID, n=self.type), self.image)
        cv2.waitKey(1)


class CollisionSensor(Sensor):
    collided: bool

    def __init__(self, vehicle, debug=False):
        super().__init__(vehicle, debug)
        self.collided = False
        colsensor = super().blueprints().find('sensor.other.collision')
        where = carla.Transform(carla.Location(x=1.5, z=0.7))
        self.setSensor(self.world().spawn_actor(colsensor, where, attach_to=self.reference()))
        self.sensor.listen(lambda collision: self.processCollison(collision))

    def processCollison(self, collision):
        self.collided = True
        print("Vehicle {id} collided!".format(id=self.vehicle.threadID))

    def isCollided(self):
        return self.collided


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