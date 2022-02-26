import glob
import threading
import os
import sys
import random
import time
import numpy as np
import cv2
import math

# from Carla doc
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla


def setMap(client):
    world = client.load_world('Town04')


def spawnMorons(client):
    world = client.get_world()
    map = world.get_map()
    spawnPoints = map.get_spawn_points()
    blueprints = world.get_blueprint_library()
    #Location(x=-8.837585, y=49.015083, z=0.281942)
    for i in range(129, 134):
        print(spawnPoints[i])
        spawnLocation = spawnPoints[i]
        veh = world.spawn_actor(blueprints.filter('model3')[0], spawnLocation)
        #time.sleep(15)
        veh.destroy()


if __name__ == '__main__':
    client = carla.Client('localhost', 2000)
    client.set_timeout(10)
    setMap(client)
    time.sleep(10)
    spawnMorons(client)