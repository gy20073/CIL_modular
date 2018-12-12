#!/usr/bin/env python

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Keyboard controlling for CARLA. Please refer to client_example.py for a simpler
# and more documented example.

"""
Welcome to CARLA manual control.
Use ARROWS or WASD keys for control.
    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
STARTING in a moment...
"""

from __future__ import print_function

import sys

sys.path.append(
    'PythonAPI/carla-0.9.0-py%d.%d-linux-x86_64.egg' % (sys.version_info.major,
                                                        sys.version_info.minor))

sys.path.append(
    'PythonAPI/carla-0.9.1-py%d.%d-linux-x86_64.egg' % (sys.version_info.major,
                                                        sys.version_info.minor))

import carla
import cv2

import copy
import argparse
import logging
import random
import time

try:
    import pygame
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_m
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_e
    from pygame.locals import K_z


except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


# the configuration begin
DELTA_POS = 0.5
WINDOW_WIDTH = 1800
WINDOW_HEIGHT = 1080
CAMERA_FOV = 120.0
CAMERA_POSITION = carla.Transform(location=carla.Location(x=0.5, z=30), rotation=carla.Rotation(roll=0, yaw=0, pitch=-90))

CAMERA_CAR_CENTER =  carla.Location(x=0.5, z=1.60)
CAMERA_CAR_ROTATION = carla.Rotation(roll=0.0, pitch=0.0, yaw=0.0)
CAMERA_CAR_POSITION = carla.Transform(location=CAMERA_CAR_CENTER, rotation=CAMERA_CAR_ROTATION)

TownName = "RFS_MAP"
output_path = "positions_file_" + TownName + ".txt"

# the configuration ends

updated = False
class CallBack():
    def __init__(self, tag, obj):
        self._tag = tag
        self._obj = obj

    def __call__(self, image):
        self._parse_image_cb(image, self._tag)

    def _parse_image_cb(self, image, tag):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = copy.deepcopy(array)

        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        sp = array.shape
        large = 30
        width = 4
        array[sp[0] // 2 - large:sp[0]//2 + large, sp[1] // 2 - width:sp[1] // 2 + width, :] = 255
        array[sp[0] // 2 - width:sp[0]//2 + width, sp[1] // 2 - large:sp[1] // 2 + large, :] = 255

        self._obj._data_buffers[self._tag] = array
        global updated
        updated = True

class CarlaGame(object):
    def __init__(self, args):
        self._client = carla.Client(args.host, args.port)
        self._client.set_timeout(2.0)
        self._display = None
        self._surface = None
        self._camera = None
        self._camera_car = None
        self._vehicle = None
        self._vehicle_yaw = 0.0
        self._autopilot_enabled = args.autopilot
        self._is_on_reverse = False
        self._spawn_new_car = False

        self._data_buffers = {}
        self._view_mode = 'aerial_camera'
        self._positions_file = None

        # initialize serialization file
        try:
            self._positions_file = open(args.positions_file, 'a+')
        except Exception as error:
            logging.error(error)

    def __del__(self):
        self._positions_file.close()

    def execute(self):
        pygame.init()
        try:
            self._display = pygame.display.set_mode(
                (WINDOW_WIDTH, WINDOW_HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
            logging.debug('pygame started')

            self._world = self._client.get_world()

            cam_blueprint = self._world.get_blueprint_library().find('sensor.camera.rgb')
            cam_blueprint.set_attribute('image_size_x', str(WINDOW_WIDTH))
            cam_blueprint.set_attribute('image_size_y', str(WINDOW_HEIGHT))
            cam_blueprint.set_attribute('fov', str(CAMERA_FOV))
            self._camera = self._world.spawn_actor(cam_blueprint, CAMERA_POSITION)
            self._camera.listen(CallBack('aerial_camera', self))


            last = time.time()
            while True:
                events = pygame.event.get()
                keys_pressed = pygame.key.get_pressed()
                for event in events:
                    if event.type == pygame.QUIT:
                        return
                self._on_loop(events, keys_pressed)
                self._on_render()
                now = time.time()
                #print(1.0 / (now-last), "Hz")
                #print(1.0 / (now-last), "Hz")
                last = now
        finally:
            pygame.quit()
            if self._camera is not None:
                self._camera.destroy()
                self._camera = None
            if self._vehicle is not None:
                self._vehicle.destroy()
                self._vehicle = None

    def _on_loop(self, events, keys_pressed):
        if self._view_mode == 'aerial_camera':
            control = self._get_keyboard_control_aerial(events, keys_pressed)
            if np.abs(control.x) < 0.01 and np.abs(control.y) < 0.01 and np.abs(control.z) < 0.01:
                pass
            else:
                self._camera.set_location(self._camera.get_location() + control)
        else:
            control = self._get_keyboard_control_vehicle(events, keys_pressed)
            self._vehicle.apply_control(control)

        if self._spawn_new_car:
            with open(output_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    sp = line.strip().split(",")
                    x, y = [float(t.strip()) for t in sp]
                    print(x, y)

        if self._spawn_new_car:
            if self._camera_car != None:
                self._camera_car.destroy()
            if self._vehicle != None:
                self._vehicle.destroy()

            vechile_blueprint =random.choice(self._world.get_blueprint_library().filter('vehicle'))
            start_transform = self._camera.get_transform()
            start_transform.location.z = 6.0
            start_transform.rotation.pitch = 0.0
            start_transform.rotation.yaw = self._vehicle_yaw
            self._vehicle = self._world.try_spawn_actor(vechile_blueprint,start_transform)

            with open(output_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    sp = line.strip().split(",")
                    x, y = [float(t.strip()) for t in sp]
                    print(x, y)
                    start_transform.location.x = x
                    start_transform.location.y = y
                    self._world.try_spawn_actor(vechile_blueprint, start_transform)

            cam_blueprint =  self._world.get_blueprint_library().find('sensor.camera.rgb')
            cam_blueprint.set_attribute('image_size_x', str(WINDOW_WIDTH))
            cam_blueprint.set_attribute('image_size_y', str(WINDOW_HEIGHT))
            cam_blueprint.set_attribute('fov', str(CAMERA_FOV))
            self._camera_car = self._world.spawn_actor(cam_blueprint, CAMERA_CAR_POSITION, attach_to=self._vehicle)
            self._camera_car.listen(CallBack('car_camera', self))
        global updated
        if updated:
            if self._view_mode in self._data_buffers:
                self._surface = pygame.surfarray.make_surface(self._data_buffers[self._view_mode].swapaxes(0, 1))
            updated = False
            updated = True
        else:
            self._surface = None

    def _get_keyboard_control_vehicle(self, input_events, keys):
        control = carla.VehicleControl()

        if keys[K_LEFT] or keys[K_a]:
            control.steer = -1.0
        if keys[K_RIGHT] or keys[K_d]:
            control.steer = 1.0
        if keys[K_UP] or keys[K_w]:
            control.throttle = 1.0
        if keys[K_DOWN] or keys[K_s]:
            control.brake = 1.0
        if keys[K_SPACE]:
            control.hand_brake = True
        if keys[K_q]:
            self._is_on_reverse = not self._is_on_reverse
        if keys[K_p]:
            self._autopilot_enabled = not self._autopilot_enabled

        for event in input_events:
            if event.type == pygame.KEYDOWN:
                if event.key == K_z:
                    self.register_position()
                if event.key == K_SPACE:
                    control.hand_brake = True
                if event.key == K_TAB:
                    if self._view_mode == 'car_camera':
                        self._view_mode = 'aerial_camera'
                    else:
                        self._view_mode = 'car_camera'

        control.reverse = self._is_on_reverse
        return control

    def _get_keyboard_control_aerial(self, input_events, keys):
        control = {'x':0, 'y':0, 'z':0}
        self._spawn_new_car = False

        if keys[K_LEFT] or keys[K_a]:
            control['y'] -= DELTA_POS
        if keys[K_RIGHT] or keys[K_d]:
            control['y'] += DELTA_POS
        if keys[K_UP] or keys[K_w]:
            control['x'] += DELTA_POS
        if keys[K_DOWN] or keys[K_s]:
            control['x'] -= DELTA_POS
        if keys[K_q]:
            control['z'] -= DELTA_POS
        if keys[K_e]:
            control['z'] += DELTA_POS

        for event in input_events:
            if event.type == pygame.KEYDOWN:
                if event.key == K_z:
                    self.register_position_aerial()
                if event.key == K_m:
                    self._vehicle_yaw = (self._vehicle_yaw + 90.0) % 360
                if event.key == K_SPACE:
                    self._spawn_new_car = True
                if event.key == K_TAB:
                    if self._view_mode == 'car_camera':
                        self._view_mode = 'aerial_camera'
                    else:
                        self._view_mode = 'car_camera'

        T = carla.Location(x=control['x'], y=control['y'], z=control['z'])

        return T

    def _on_render(self):
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))
        pygame.display.flip()


    def register_position_aerial(self):
        location = self._camera.get_location()
        self._positions_file.write(
            "{}, {}\n".format(location.x, location.y))
        print(location.x, location.y)
        print("position recorded aerial")

    def register_position(self):
        vehicle_location = self._vehicle.get_location()
        vehicle_rotation = self._vehicle.get_transform().rotation
        self._positions_file.write(
            "{}, {}, {}, {}, {}, {}\n".format(vehicle_location.x, vehicle_location.y, vehicle_location.z, vehicle_rotation.pitch,
                                          vehicle_rotation.roll, vehicle_rotation.yaw))
        print("position recorded")


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--positions_file',
        default=output_path,
        help='Filename to store positions')
    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    while True:
        try:

            game = CarlaGame(args)
            game.execute()
            break

        except Exception as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
