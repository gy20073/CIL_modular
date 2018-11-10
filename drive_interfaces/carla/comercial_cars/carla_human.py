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
        sys.path.append('drive_interfaces/carla/carla_client_090/carla-0.9.0-py2.7-linux-x86_64.egg')
    else:
        sys.path[0:0]=['/scratch/yang/aws_data/carla_auto2/PythonAPI/carla-0.9.0-py2.7-linux-x86_64.egg']
        print(sys.path)
    import carla
    from carla import Client as CarlaClient
    from carla import VehicleControl as VehicleControl
    from scenario_manager import ScenarioManager

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
        self._camera_center = None
        self._spectator = None
        # (last) images store for several cameras


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


        self._scenario_manager = None

        self._sensor_list = []

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

    def start(self):
        if __CARLA_VERSION__ == '0.8.X':
            self.carla = CarlaClient(self._host, int(self._port), timeout=120)
            self.carla.connect()
        else:
            self.carla = CarlaClient(self._host, int(self._port))
            self.carla.set_timeout(2000)

            opts_scenario = {}

            if self._autopilot:
                opts_scenario['weather'] = self._weather_list[int(self._driver_conf.weather) - 1]
            else:
                opts_scenario['weather'] = None
                # select one of the random starting points previously selected
                start_positions = np.loadtxt(self._driver_conf.positions_file, delimiter=',')
                if len(start_positions.shape) == 1:
                    start_positions =  start_positions.reshape(1, len(start_positions))
                opts_scenario['start_positions'] = start_positions


            self._scenario_manager = ScenarioManager(world=self.carla.get_world(), opts=opts_scenario)

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
               self._scenario_manager.__del__()

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
            # set scenario
            self._scenario_manager.set_scenario(id='SCENARIO_RANDOM')



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
                  or self._stucked_counter > 150:
                    if self._stucked_counter > 150:
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
                or self._stucked_counter > 250 \
                or np.abs(self._vehicle.get_vehicle_control().steer) > 0.95:
                    # TODO intersection other lane is not available, so omit from the condition right now
                    if self._stucked_counter > 250:
                        reset_because_stuck = True
                    else:
                        reset_because_stuck = False
                    if np.abs(self._vehicle.get_vehicle_control().steer) > 0.95:
                        print("reset because of large steer")

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
                control = self._vehicle.get_vehicle_control()

        print('[Throttle = {}] [Steering = {}] [Brake = {}]'.format(control.throttle, control.steer, control.brake))
        return control

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
            dict_data = self._scenario_manager.get_sync_sensor_data()
            measurements = dict_data['measurements']
            sensor_data = dict_data['sensor_data']
            direction = self._current_command


        return measurements, sensor_data, direction

    def act(self, control):
        if __CARLA_VERSION__ == '0.8.X':
            self.carla.send_control(control)
        else:
            self._scenario_manager.hero_apply_control(control)

            # location = self._vehicle.get_location()
            # location.z = 200.0
            # rotation = self._vehicle.get_transform().rotation
            # rotation.pitch = -90.0
            # rotation.yaw = 0
            # rotation.roll = -0.07

            #print('>>>>> x={}, y={}, z={}, pitch={}, yaw={}, roll={}'.format(location.x, location.y, location.z, rotation.pitch, rotation.yaw, rotation.roll))
            #self._spectator.set_transform(carla.Transform(location=location, rotation=rotation))
