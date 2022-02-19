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

client = carla.Client('localhost', 2000)
client.set_timeout(10)
world = client.get_world()
l = world.get_actors()
for a in l:
    a.destroy()