import glob
import os
import sys
import carla
import random
import time
import numpy as np
import cv2
## global constants
IM_WIDTH = 640
IM_HEIGHT = 480

actor_list = []

def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    cv2.imshow("Camera", i3)
    cv2.waitKey(1)
    return i3/255.0


def runCarla():
    try:
        sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    except IndexError:
        pass
    client = carla.Client('localhost', 2000)
    client.set_timeout(4.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    car = random.choice(blueprint_library.filter('vehicle'))
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(car, spawn_point)
    print("Car spawned")
    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
    actor_list.append(vehicle)
# get the blueprint for this sensor
    camera = blueprint_library.find('sensor.camera.rgb')
# change the dimensions of the image
    camera.set_attribute('image_size_x', f'{IM_WIDTH}')
    camera.set_attribute('image_size_y', f'{IM_HEIGHT}')
    camera.set_attribute('fov', '110')

    spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
    sensor = world.spawn_actor(camera, spawn_point, attach_to=vehicle)
    actor_list.append(sensor)

    # do something with this sensor
    sensor.listen(lambda data: process_img(data))
    time.sleep(15)

    for actor in actor_list:
        actor.destroy()
        print("Car destroyed")
        actor_list.remove(vehicle)
