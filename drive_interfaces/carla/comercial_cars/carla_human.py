import threading
import weakref
import cv2
import copy
import sys, os
import pygame, io, math, time
import random
from configparser import ConfigParser
from configparser import SafeConfigParser
from datetime import datetime
import numpy as np
import math as m
from collections import namedtuple

__CARLA_VERSION__ = os.getenv('CARLA_VERSION', '0.8.X')
if __CARLA_VERSION__ == '0.8.X':
    from carla.planner.planner import Planner
    from carla.client import VehicleControl
    #from carla.client import make_carla_client
    from carla.client import CarlaClient
else:
    if __CARLA_VERSION__ == '0.9.X':
        sys.path.append('drive_interfaces/carla/carla_client_090/carla-0.9.1-py2.7-linux-x86_64.egg')
    else:
        sys.path[0:0]=['/scratch/yang/aws_data/carla_auto2/PythonAPI/carla-0.9.1-py2.7-linux-x86_64.egg']
        print(sys.path)
    import carla
    from carla import Client as CarlaClient
    from carla import VehicleControl as VehicleControl
    from Navigation.roaming_agent import *

from driver import Driver

sldist = lambda c1, c2: math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)

def v3d_to_array(v3d):
    return [v3d.x, v3d.y, v3d.z]

def find_valid_episode_position(positions, planner):
    while True:
        print("a new trail to find a start and end position")
        index_start = np.random.randint(len(positions))
        start_pos = positions[index_start]
        # The test position func means is_away_from_intersection
        if not planner.test_position(v3d_to_array(start_pos.location)):
            continue

        index_goal = np.random.randint(len(positions))
        goals_pos = positions[index_goal]
        print((' TESTING start and end location (', index_start, ',', index_goal, ')'))
        if not planner.test_position(v3d_to_array(goals_pos.location)):
            continue

        dist = sldist([start_pos.location.x, start_pos.location.y], [goals_pos.location.x, goals_pos.location.y])
        if dist < 250.0:
            print("continue because distance too close", dist, start_pos, goals_pos)
            continue

        if planner.is_there_posible_route(v3d_to_array(start_pos.location), v3d_to_array(start_pos.orientation),
                                          v3d_to_array(goals_pos.location), v3d_to_array(goals_pos.orientation)):
            break

    return index_start, index_goal

data_buffer_lock = threading.Lock()
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

        '''
        if self._tag == "CameraMiddle":
            while self._obj.update_once == True:
                time.sleep(0.01)
        '''

        data_buffer_lock.acquire()
        self._obj._data_buffers[self._tag] = array
        data_buffer_lock.release()
        '''
        if self._tag == "CameraMiddle":
            self._obj.update_once = True
        '''

collision_lock = threading.Lock()
class CollisionSensor(object):
    def __init__(self, parent_actor, obj):
        self._obj = obj
        self.sensor = None
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = ' '.join(event.other_actor.type_id.replace('_', '.').title().split('.')[0:1])
        #'Collision with %r' % actor_type
        collision_lock.acquire()
        self._obj._collision_events.append(actor_type)
        collision_lock.release()


class CarlaHuman(Driver):
    def __init__(self, driver_conf):
        Driver.__init__(self)
        # some behaviors
        self._autopilot = driver_conf.autopilot
        self._reset_period = driver_conf.reset_period # those reset period are in the actual system time, not in simulation time
        self._goal_reaching_threshold = 3
        self.use_planner = driver_conf.use_planner
        # we need the planner to find a valid episode, so we initialize one no matter what

        self._world = None
        self._vehicle = None
        self._agent_autopilot = None
        self._camera_center = None
        self._spectator = None
        # (last) images store for several cameras
        self._data_buffers = dict()
        self.update_once = False
        self._collision_events = []
        self.collision_sensor = None

        if __CARLA_VERSION__ == '0.8.X':
            self.planner = Planner(driver_conf.city_name)
        else:
            self.planner = None
            self.use_planner = False

        # resources
        self._host = driver_conf.host
        self._port = driver_conf.port

        # various config files
        self._driver_conf = driver_conf
        self._config_path = driver_conf.carla_config

        # some initializations
        self._straight_button = False
        self._left_button = False
        self._right_button = False

        self._rear = False
        self._recording = False
        self._skiped_frames = 20
        self._stucked_counter = 0

        self._prev_time = datetime.now()
        self._episode_t0 = datetime.now()

        self._vehicle_prev_location = namedtuple("vehicle", "x y z")
        self._vehicle_prev_location.x = 0.0
        self._vehicle_prev_location.y = 0.0
        self._vehicle_prev_location.z = 0.0

        self._camera_left = None
        self._camera_right = None
        self._camera_center = None

        self._actor_list = []

        self._sensor_list = []
        self._weather_list = ['ClearNoon', 'CloudyNoon', 'WetNoon', 'WetCloudyNoon',
                              'MidRainyNoon', 'HardRainNoon', 'SoftRainNoon', 'ClearSunset',
                              'CloudySunset', 'WetSunset', 'WetCloudySunset', 'MidRainSunset',
                              'HardRainSunset', 'SoftRainSunset']

        self._current_weather = 4

        self._current_command = 2.0

        # steering wheel
        self._steering_wheel_flag = True

        if self._steering_wheel_flag:
            self._is_on_reverse = False
            self._control = VehicleControl()
            self._parser = SafeConfigParser()
            self._parser.read('wheel_config.ini')
            self._steer_idx = int(self._parser.get('G29 Racing Wheel', 'steering_wheel'))
            self._throttle_idx = int(self._parser.get('G29 Racing Wheel', 'throttle'))
            self._brake_idx = int(self._parser.get('G29 Racing Wheel', 'brake'))
            self._reverse_idx = int(self._parser.get('G29 Racing Wheel', 'reverse'))
            self._handbrake_idx = int(self._parser.get('G29 Racing Wheel', 'handbrake'))
        self.last_timestamp = lambda x: x
        self.last_timestamp.elapsed_seconds = 0.0
        self.last_timestamp.delta_seconds = 0.2


    def start(self):
        if __CARLA_VERSION__ == '0.8.X':
            self.carla = CarlaClient(self._host, int(self._port), timeout=120)
            self.carla.connect()
        else:
            self.carla = CarlaClient(self._host, int(self._port))
            self.carla.set_timeout(2000)

        self._reset()

        if not self._autopilot:
            pygame.joystick.init()

            joystick_count = pygame.joystick.get_count()
            if joystick_count > 1:
                print("Please Connect Just One Joystick")
                raise ValueError()

            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()

    def __del__(self):
        if hasattr(self, 'carla'):
            print("destructing the connection")
            if __CARLA_VERSION__ == '0.8.X':
                self.carla.disconnect()
            else:
                # destroy old actors
                print('destroying actors')
                if len(self._actor_list) > 0:
                    for actor in self._actor_list:
                        actor.destroy()
                self._actor_list = []
                print('done.')

                if self._vehicle is not None:
                    self._vehicle.destroy()
                    self._vehicle = None
                if self._camera_center is not None:
                    self._camera_center.destroy()
                    self._camera_center = None
                if self._camera_left is not None:
                    self._camera_left.destroy()
                    self._camera_left = None
                if self._camera_right is not None:
                    self._camera_right.destroy()
                    self._camera_right = None
                if self.collision_sensor is not None:
                    self.collision_sensor.sensor.destroy()
                    self.collision_sensor = None


                    #  pygame.quit()
            # if self._camera is not None:
            #     self._camera.destroy()
            #     self._camera = None
            # if self._vehicle is not None:
            #     self._vehicle.destroy()
            #     self._vehicle = None

    def try_spawn_random_vehicle_at(self, blueprints, transform, auto_drive=True):
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        vehicle = self._world.try_spawn_actor(blueprint, transform)
        if vehicle is not None:
            self._actor_list.append(vehicle)
            if auto_drive:
                vehicle.set_autopilot()
            #print('spawned %r at %s' % (vehicle.type_id, transform.location))
            return True
        return False

    def get_parking_locations(self, filename, z_default=0.0, random_perturb=False):
        with open(filename, "r") as f:
            lines = f.readlines()
            ans = []
            for line in lines:
                x, y, yaw = [float(v.strip()) for v in line.split(",")]
                if random_perturb:
                    x += np.random.normal(0, scale=self._driver_conf.extra_explore_location_std)
                    y += np.random.normal(0, scale=self._driver_conf.extra_explore_location_std)
                    yaw+=np.random.normal(0, scale=self._driver_conf.extra_explore_yaw_std)

                ans.append(carla.Transform(location=carla.Location(x=x, y=y, z=z_default),
                                           rotation=carla.Rotation(roll=0, pitch=0, yaw=yaw)))
        return ans

    def print_transform(self, t):
        print(t.location.x, t.location.y, t.location.z)
        print(t.rotation.roll, t.rotation.pitch, t.rotation.yaw)

    def _reset(self):

        self._start_time = time.time()
        self._episode_t0 = datetime.now()


        if __CARLA_VERSION__ == '0.8.X':
            # create the carla config based on template and the params passed in
            config = ConfigParser()
            config.optionxform = str
            config.read(self._config_path)
            config.set('CARLA/LevelSettings', 'NumberOfVehicles',    self._driver_conf.cars)
            config.set('CARLA/LevelSettings', 'NumberOfPedestrians', self._driver_conf.pedestrians)
            config.set('CARLA/LevelSettings', 'WeatherId',           self._driver_conf.weather)
            output = io.StringIO()
            config.write(output)
            scene_descriptions = self.carla.load_settings(output.getvalue())

            # based on the scene descriptions, find the start and end positions
            self.positions = scene_descriptions.player_start_spots
            # the episode_config saves [start_index, end_index]
            self.episode_config = find_valid_episode_position(self.positions, self.planner)

            self.carla.start_episode(self.episode_config[0])
            print('RESET ON POSITION ', self.episode_config[0], ", the target location is: ", self.episode_config[1])

        else:
            # destroy old actors
            print('destroying actors')
            for actor in self._actor_list:
                actor.destroy()
            self._actor_list = []
            print('done.')

            # TODO: spawn pedestrains
            # TODO: spawn more vehicles

            if self._autopilot:
                self._current_weather = self._weather_list[int(self._driver_conf.weather)-1]
            else:
                self._current_weather = random.choice(self._weather_list)
            # select one of the random starting points previously selected
            start_positions = np.loadtxt(self._driver_conf.positions_file, delimiter=',')
            if len(start_positions.shape) == 1:
                start_positions = start_positions.reshape(1, len(start_positions))


            # TODO: Assign random position from file
            WINDOW_WIDTH = 768
            WINDOW_HEIGHT = 576
            CAMERA_FOV = 103.0


            CAMERA_CENTER_T = carla.Location(x=0.7, y=-0.0, z=1.60)
            CAMERA_LEFT_T = carla.Location(x=0.7, y=-0.4, z=1.60)
            CAMERA_RIGHT_T = carla.Location(x=0.7, y=0.4, z=1.60)

            CAMERA_CENTER_ROTATION = carla.Rotation(roll=0.0, pitch=0.0, yaw=0.0)
            CAMERA_LEFT_ROTATION = carla.Rotation(roll=0.0, pitch=0.0, yaw=-45.0)
            CAMERA_RIGHT_ROTATION = carla.Rotation(roll=0.0, pitch=0.0, yaw=45.0)

            CAMERA_CENTER_TRANSFORM = carla.Transform(location=CAMERA_CENTER_T, rotation=CAMERA_CENTER_ROTATION)
            CAMERA_LEFT_TRANSFORM = carla.Transform(location=CAMERA_LEFT_T, rotation=CAMERA_LEFT_ROTATION)
            CAMERA_RIGHT_TRANSFORM = carla.Transform(location=CAMERA_RIGHT_T, rotation=CAMERA_RIGHT_ROTATION)


            self._world = self.carla.get_world()


            # add traffic
            blueprints_vehi = self._world.get_blueprint_library().filter('vehicle.*')
            blueprints_vehi = [x for x in blueprints_vehi if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints_vehi = [x for x in blueprints_vehi if not x.id.endswith('isetta')]

            # @todo Needs to be converted to list to be shuffled.
            spawn_points = list(self._world.get_map().get_spawn_points())
            random.shuffle(spawn_points)

            print('found %d spawn points.' % len(spawn_points))

            # TODO: debug change 50 to 0
            count = 0

            for spawn_point in spawn_points:
                if self.try_spawn_random_vehicle_at(blueprints_vehi, spawn_point):
                    count -= 1
                if count <= 0:
                    break
            while count > 0:
                time.sleep(0.5)
                if self.try_spawn_random_vehicle_at(blueprints_vehi, random.choice(spawn_points)):
                    count -= 1
            # end traffic addition!

            # begin parking addition
            if hasattr(self._driver_conf, "parking_position_file") and self._driver_conf.parking_position_file is not None:
                parking_points = self.get_parking_locations(self._driver_conf.parking_position_file)
                random.shuffle(parking_points)
                print('found %d parking points.' % len(parking_points))
                count = 50

                for spawn_point in parking_points:
                    self.try_spawn_random_vehicle_at(blueprints_vehi, spawn_point, False)
                    count -= 1
                    if count <= 0:
                        break
            # end of parking addition

            blueprints = self._world.get_blueprint_library().filter('vehicle')
            vechile_blueprint = [e for i, e in enumerate(blueprints) if e.id == 'vehicle.lincoln.mkz2017'][0]


            if self._vehicle == None or self._autopilot:
                if self._autopilot and self._vehicle is not None:
                    self._vehicle.destroy()
                    self._vehicle = None

                while self._vehicle == None:
                    if self._autopilot:
                        # from designated points
                        if hasattr(self._driver_conf, "extra_explore_prob") and random.random() < self._driver_conf.extra_explore_prob:
                            extra_positions = self.get_parking_locations(self._driver_conf.extra_explore_position_file,
                                                                         z_default=3.0, random_perturb=True)
                            print("spawning hero vehicle from the extra exploration")
                            START_POSITION = random.choice(extra_positions)
                        else:
                            START_POSITION = random.choice(spawn_points)
                    else:
                        random_position = start_positions[np.random.randint(start_positions.shape[0]), :]
                        START_POSITION = carla.Transform(
                            carla.Location(x=random_position[0], y=random_position[1], z=random_position[2] + 1.0),
                            carla.Rotation(pitch=random_position[3], roll=random_position[4], yaw=random_position[5]))

                    self._vehicle = self._world.try_spawn_actor(vechile_blueprint, START_POSITION)
            else:
                if self._autopilot:
                    # from designated points
                    START_POSITION = random.choice(spawn_points)
                else:
                    random_position = start_positions[np.random.randint(start_positions.shape[0]), :]
                    START_POSITION = carla.Transform(
                        carla.Location(x=random_position[0], y=random_position[1], z=random_position[2] + 1.0),
                        carla.Rotation(pitch=random_position[3], roll=random_position[4], yaw=random_position[5]))

                self._vehicle.set_transform(START_POSITION)

            if self._autopilot:
                # Nope: self._vehicle.set_autopilot()
                self._agent_autopilot = RoamingAgent(self._vehicle)

            if self.collision_sensor is not None:
                self.collision_sensor.sensor.destroy()
            self.collision_sensor = CollisionSensor(self._vehicle, self)

            # set weather
            weather = getattr(carla.WeatherParameters, self._current_weather)
            self._vehicle.get_world().set_weather(weather)

            self._spectator = self._world.get_spectator()
            cam_blueprint = self._world.get_blueprint_library().find('sensor.camera.rgb')

            cam_blueprint.set_attribute('image_size_x', str(WINDOW_WIDTH))
            cam_blueprint.set_attribute('image_size_y', str(WINDOW_HEIGHT))
            cam_blueprint.set_attribute('fov', str(CAMERA_FOV))


            if self._camera_center is not None:
                self._camera_center.destroy()
                self._camera_left.destroy()
                self._camera_right.destroy()
                self._camera_center = None


            if self._camera_center == None:
                self._camera_center = self._world.spawn_actor(cam_blueprint, CAMERA_CENTER_TRANSFORM, attach_to=self._vehicle)
                self._camera_left = self._world.spawn_actor(cam_blueprint, CAMERA_LEFT_TRANSFORM, attach_to=self._vehicle)
                self._camera_right = self._world.spawn_actor(cam_blueprint, CAMERA_RIGHT_TRANSFORM, attach_to=self._vehicle)

                self._camera_center.listen(CallBack('CameraMiddle', self))
                self._camera_left.listen(CallBack('CameraLeft', self))
                self._camera_right.listen(CallBack('CameraRight', self))

            # spectator server camera
            self._spectator = self._world.get_spectator()



        self._skiped_frames = 0
        self._stucked_counter = 0

    def get_recording(self):
        if self._autopilot:
            # debug: 0 for debugging
            if self._skiped_frames >= 20:
                return True
            else:
                self._skiped_frames += 1
                return False

        else:
            '''
            if (self.joystick.get_button(8)):
                self._recording = True
            if (self.joystick.get_button(9)):
                self._recording = False
            '''
            if (self.joystick.get_button(6)):
                self._recording = True
                print("start recording!!!!!!!!!!!!!!!!!!!!!!!!1")
            if (self.joystick.get_button(7)):
                self._recording = False
                print("end recording!!!!!!!!!!!!!!!!!!!!!!!!1")
            return self._recording

    def get_reset(self):
        if self._autopilot:
            if __CARLA_VERSION__ == '0.8.X':
                # increase the stuck detector if conditions satisfy
                if self._latest_measurements.player_measurements.forward_speed < 0.1:
                    self._stucked_counter += 1
                else:
                    self._stucked_counter = 0

                # if within auto pilot, reset if long enough or has collisions
                if time.time() - self._start_time > self._reset_period \
                  or self._latest_measurements.player_measurements.collision_vehicles    > 0.0 \
                  or self._latest_measurements.player_measurements.collision_pedestrians > 0.0 \
                  or self._latest_measurements.player_measurements.collision_other       > 0.0 \
                  or (self._latest_measurements.player_measurements.intersection_otherlane > 0.0 and self._latest_measurements.player_measurements.autopilot_control.steer < -0.99) \
                  or self._stucked_counter > 250:
                    if self._stucked_counter > 250:
                        reset_because_stuck = True
                    else:
                        reset_because_stuck = False

                    # TODO: commenting out this for debugging issue
                    self._reset()

                    if reset_because_stuck:
                        print("resetting because getting stucked.....")
                        return True
            else:
                # TODO: implement the collision detection algorithm, based on the new API
                if self.last_estimated_speed < 0.1:
                    self._stucked_counter += 1
                else:
                    self._stucked_counter = 0

                if time.time() - self._start_time > self._reset_period \
                or self._last_collided \
                or self._stucked_counter > 250:
                #or np.abs(self._vehicle.get_vehicle_control().steer) > 0.95:
                #or np.abs(self._vehicle.get_vehicle_control().brake) > 1:
                    # TODO intersection other lane is not available, so omit from the condition right now
                    if self._stucked_counter > 250:
                        reset_because_stuck = True
                    else:
                        reset_because_stuck = False
                    if np.abs(self._vehicle.get_vehicle_control().steer) > 0.95:
                        #print("reset because of large steer")
                        pass
                    if self._last_collided:
                        print("reset becuase collision")

                    self._reset()

                    if reset_because_stuck:
                        print("resetting because getting stucked.....")
                        return True
        else:
            pass

        return False

    def get_waypoints(self):
        # TODO: waiting for German Ros to expose the waypoints
        wp1 = [1.0, 1.0]
        wp2 = [2.0, 2.0]
        return [wp1, wp2]

    def action_joystick(self):
        # joystick
        steering_axis = self.joystick.get_axis(0)
        acc_axis = self.joystick.get_axis(2)
        brake_axis = self.joystick.get_axis(5)
        # print("axis 0 %f, axis 2 %f, axis 3 %f" % (steering_axis, acc_axis, brake_axis))

        if (self.joystick.get_button(3)):
            self._rear = True
        if (self.joystick.get_button(2)):
            self._rear = False

        control = VehicleControl()
        control.steer = steering_axis
        control.throttle = (acc_axis + 1) / 2.0
        control.brake = (brake_axis + 1) / 2.0
        if control.brake < 0.001:
            control.brake = 0.0
        control.hand_brake = 0
        control.reverse = self._rear

        control.steer -= 0.0822

        #print("steer %f, throttle %f, brake %f" % (control.steer, control.throttle, control.brake))
        pygame.event.pump()

        return control

    def action_steering_wheel(self, jsInputs, jsButtons):
        control = VehicleControl()

        # Custom function to map range of inputs [1, -1] to outputs [0, 1] i.e 1 from inputs means nothing is pressed
        # For the steering, it seems fine as it is
        K1 = 1.0  # 0.55
        steerCmd = K1 * math.tan(1.1 * jsInputs[self._steer_idx])

        K2 = 1.6  # 1.6
        throttleCmd = K2 + (2.05 * math.log10(-0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
        if throttleCmd <= 0:
            throttleCmd = 0
        elif throttleCmd > 1:
            throttleCmd = 1

        brakeCmd = 1.6 + (2.05 * math.log10(-0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92
        if brakeCmd <= 0:
            brakeCmd = 0
        elif brakeCmd > 1:
            brakeCmd = 1

        #print("Steer Cmd, ", steerCmd, "Brake Cmd", brakeCmd, "ThrottleCmd", throttleCmd)
        control.steer = steerCmd
        control.brake = brakeCmd
        control.throttle = throttleCmd
        toggle = jsButtons[self._reverse_idx]

        if toggle == 1:
            self._is_on_reverse += 1
        if self._is_on_reverse % 2 == 0:
            control.reverse = False
        if self._is_on_reverse > 1:
            self._is_on_reverse = True

        if self._is_on_reverse:
            control.reverse = True

        control.hand_brake = False  # jsButtons[self.handbrake_idx]

        return control


    def compute_action(self, sensor, speed):
        if not self._autopilot:
            # get pygame input
            for event in pygame.event.get():
                # Possible joystick actions: JOYAXISMOTION JOYBALLMOTION JOYBUTTONDOWN
                # JOYBUTTONUP JOYHATMOTION
                if event.type == pygame.JOYBUTTONDOWN:
                    if event.__dict__['button'] == 0:
                        self._current_command = 2.0
                    if event.__dict__['button'] == 1:
                        self._current_command = 3.0
                    if event.__dict__['button'] == 2:
                        self._current_command = 4.0
                    if event.__dict__['button'] == 3:
                        self._current_command = 5.0
                    if event.__dict__['button'] == 23:
                        self._current_command = 0.0
                    if event.__dict__['button'] == 4:
                        self._reset()
                        return VehicleControl()
                if event.type == pygame.JOYBUTTONUP:
                    self._current_command = 2.0


            #pygame.event.pump()
            numAxes = self.joystick.get_numaxes()
            jsInputs = [float(self.joystick.get_axis(i)) for i in range(numAxes)]
            # print (jsInputs)
            jsButtons = [float(self.joystick.get_button(i)) for i in range(self.joystick.get_numbuttons())]

            if self._steering_wheel_flag:
                control = self.action_steering_wheel(jsInputs, jsButtons)
            else:
                control = self.action_joystick()

        else:
            if __CARLA_VERSION__ == '0.8.X':
                # This relies on the calling of get_sensor_data, otherwise self._latest_measurements are not filled
                control = self._latest_measurements.player_measurements.autopilot_control
                print('[Throttle = {}] [Steering = {}] [Brake = {}]'.format(control.throttle, control.steer, control.brake))
            else:
                if self._world.wait_for_tick(10.0):
                    control, self._current_command = self._agent_autopilot.run_step()

        print('[Throttle = {}] [Steering = {}] [Brake = {}]'.format(control.throttle, control.steer, control.brake))
        return control


    def estimate_speed(self):
        vel = self._vehicle.get_velocity()
        speed = m.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

        # speed in m/s
        return speed

    def get_sensor_data(self, goal_pos=None, goal_ori=None):
        if __CARLA_VERSION__ == '0.8.X':
            # return the latest measurement and the next direction
            measurements, sensor_data = self.carla.read_data()
            self._latest_measurements = measurements

            if self.use_planner:
                player_data = measurements.player_measurements
                pos = [player_data.transform.location.x,
                       player_data.transform.location.y,
                       0.22]
                ori = [player_data.transform.orientation.x,
                       player_data.transform.orientation.y,
                       player_data.transform.orientation.z]

                if sldist([player_data.transform.location.x,
                           player_data.transform.location.y],
                          [self.positions[self.episode_config[1]].location.x,
                           self.positions[self.episode_config[1]].location.y]) < self._goal_reaching_threshold:
                    self._reset()

                direction = self.planner.get_next_command(pos, ori,
                                                          [self.positions[self.episode_config[1]].location.x,
                                                           self.positions[self.episode_config[1]].location.y,
                                                           0.22],
                                                          (1, 0, 0))
            else:
                direction = 2.0
        else:
            self.last_timestamp.elapsed_seconds += 0.2
            #self.last_timestamp = self.carla.get_world().wait_for_tick(30.0)
            #print(timestamp.delta_seconds, "delta seconds")

            #while self.update_once == False:
            #    time.sleep(0.01)

            self.last_estimated_speed = self.estimate_speed()

            data_buffer_lock.acquire()
            sensor_data = copy.deepcopy(self._data_buffers)
            data_buffer_lock.release()

            #self.update_once = False


            collision_lock.acquire()
            colllision_event = self._collision_events
            self._last_collided = len(colllision_event) > 0
            self._collision_events = []
            collision_lock.release()

            if len(colllision_event)>0:
                print(colllision_event)
            # TODO: make sure those events are actually valid
            if 'Static' in colllision_event:
                collision_other = 1.0
            else:
                collision_other = 0.0
            if "Vehicles" in colllision_event:
                collision_vehicles = 1.0
            else:
                collision_vehicles = 0.0
            if "Pedestrians" in colllision_event:
                collision_pedestrians = 1.0
            else:
                collision_pedestrians = 0.0


            #current_ms_offset = int(math.ceil((datetime.now() - self._episode_t0).total_seconds() * 1000))
            # TODO: get a gametime stamp, instead of os timestamp
            #current_ms_offset = int(self.carla.get_timestamp().elapsed_seconds * 1000)
            #print(current_ms_offset, "ms offset")
            current_ms_offset = self.last_timestamp.elapsed_seconds * 1000

            second_level = namedtuple('second_level', ['forward_speed', 'transform', 'collision_other', 'collision_pedestrians', 'collision_vehicles'])
            transform = namedtuple('transform', ['location', 'orientation'])
            loc = namedtuple('loc', ['x', 'y'])
            ori = namedtuple('ori', ['x', 'y', 'z'])
            Meas = namedtuple('Meas', ['player_measurements', 'game_timestamp'])

            v_transform = self._vehicle.get_transform()
            measurements = Meas(
                                second_level(self.last_estimated_speed,
                                             transform(loc(v_transform.location.x,
                                                           v_transform.location.y),
                                                       ori(v_transform.rotation.pitch,
                                                           v_transform.rotation.roll,
                                                           v_transform.rotation.yaw)),
                                             collision_other,
                                             collision_pedestrians,
                                             collision_vehicles),
                                current_ms_offset)
            direction = self._current_command

            #print('[Speed = {} Km/h] [Direction = {}]'.format(measurements.player_measurements.forward_speed, direction))
        #print(">>>>> planner output direction: ", direction)

        return measurements, sensor_data, direction

    def act(self, control):
        if __CARLA_VERSION__ == '0.8.X':
            self.carla.send_control(control)
        else:
            self._vehicle.apply_control(control)