#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from enum import Enum
import random

from Navigation.controller import *
from Tools.misc import *

import global_vars

class ROAD_OPTIONS(Enum):
    """
    ROAD_OPTION represents the possible topological configurations when moving from a segment of lane to other.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4

class LocalPlanner(object):
    """
    LocalPlanner implements the basic behavior of following a trajectory of waypoints that is generated on-the-fly.
    The low-level motion of the vehicle is computed by using two PID controllers, one is used for the lateral control
    and the other for the longitudinal control (cruise speed).

    When multiple paths are available (intersections) this local planner makes a random choice.
    """

    # minimum distance to target waypoint as a percentage (e.g. within 90% of total distance)
    MIN_DISTANCE_PERCENTAGE = 0.9

    def __init__(self, vehicle, opt_dict={}):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param opt_dict: dictionary of arguments with the following semantics:
            dt -- time difference between physics control in seconds. This is typically fixed from server side
                  using the arguments -benchmark -fps=F . In this case dt = 1/F

            target_speed -- desired cruise speed in Km/h

            sampling_radius -- search radius for next waypoints in seconds: e.g. 0.5 seconds ahead

            lateral_control_dict -- dictionary of arguments to setup the lateral PID controller
                                    {'K_P':, 'K_D':, 'K_I':, 'dt'}

            longitudinal_control_dict -- dictionary of arguments to setup the longitudinal PID controller
                                        {'K_P':, 'K_D':, 'K_I':, 'dt'}
        """
        self.count_down = 0
        self._vehicle = vehicle
        self._map = self._vehicle.get_world().get_map()

        self._dt = None
        self._target_speed = None
        self._sampling_radius = None
        self._min_distance = None
        self._current_waypoint = None
        self._target_road_option = None
        self._next_waypoints = None
        self._vehicle_controller = None
        self._waypoints_queue = deque(maxlen=200) # queue with tuples of (waypoint, ROAD_OPTIONS)

        #
        self.init_controller(opt_dict)

    def __del__(self):
        self._vehicle.destroy()
        print("Destroying ego-vehicle!")

    def init_controller(self, opt_dict):
        """
        Controller initialization.

        :param opt_dict: dictionary of arguments.
        :return:
        """
        # default params
        self._dt = 1.0 / 20.0
        self._target_speed = 20.0  # Km/h
        self._sampling_radius = self._target_speed * 0.5 / 3.6 # 0.5 seconds horizon
        self._min_distance = self._sampling_radius * self.MIN_DISTANCE_PERCENTAGE
        #args_lateral_dict = {'K_P': 1.9, 'K_D': 0.0, 'K_I': 1.4, 'dt': self._dt}
        args_lateral_dict = {'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.0, 'dt': self._dt}
        args_longitudinal_dict = {'K_P': 1.0, 'K_D': 0, 'K_I': 0.0, 'dt': self._dt}

        # parameters overload
        if 'dt' in opt_dict:
            self._dt = opt_dict['dt']
        if 'target_speed' in opt_dict:
            self._target_speed = opt_dict['target_speed']
        if 'sampling_radius' in opt_dict:
            self._sampling_radius = self._target_speed * opt_dict['sampling_radius'] / 3.6
        if 'lateral_control_dict' in opt_dict:
            args_lateral_dict = opt_dict['lateral_control_dict']
        if 'longitudinal_control_dict' in opt_dict:
            args_longitudinal_dict = opt_dict['longitudinal_control_dict']

        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self._vehicle_controller = VehiclePIDController(self._vehicle,
                                                        args_lateral=args_lateral_dict,
                                                        args_longitudinal=args_longitudinal_dict)

        # vehicles need to be pre activated in order to compensate for the "manual gear" problem
        #self._vehicle_controller.warmup()
        # it seems like in 0.9.5 we don't have such a problem

        # compute initial waypoints

        loc = self._vehicle.get_location()
        print("initial location is ", loc.x, loc.y)

        cw = self._current_waypoint
        print("current waypoint druing init is ", cw.transform.location.x, cw.transform.location.y)

        self._waypoints_queue.append((self._current_waypoint.next(self._sampling_radius)[0], ROAD_OPTIONS.LANEFOLLOW, None))
        tw = self._waypoints_queue[0][0]
        print("target waypoint during init is ", tw.transform.location.x, tw.transform.location.y)


        self._target_road_option = ROAD_OPTIONS.LANEFOLLOW

        # fill waypoint trajectory queue
        self._compute_next_waypoints(k=200)

    def set_speed(self, speed):
        """
        Request new target speed.

        :param speed: new target speed in Km/h
        :return:
        """
        self._target_speed = speed

    def _compute_next_waypoints(self, k=1):
        """
        Add new waypoints to the trajectory queue.

        :param k: how many waypoints to compute
        :return:
        """
        # check we do not overflow the queue
        available_entries = self._waypoints_queue.maxlen - len(self._waypoints_queue)
        k = min(available_entries, k)

        for i in range(k):
            last_waypoint = self._waypoints_queue[-1][0]
            next_waypoints = list(last_waypoint.next(self._sampling_radius))

            diff_angle = None
            if len(next_waypoints) == 1:
                # only one option available ==> lanefollowing
                next_waypoint = next_waypoints[0]
                road_option = ROAD_OPTIONS.LANEFOLLOW
                if self.count_down > 0:
                    # override it with the road option value
                    road_option = self.count_down_value
                    self.count_down -= 1
                non_follow_event_happened = False
            else:
                self.count_down = 0
                # random choice between the possible options
                road_options_list = retrieve_options(next_waypoints, last_waypoint)
                road_option, diff_angle = random.choice(road_options_list)
                next_waypoint = next_waypoints[road_options_list.index((road_option, diff_angle))]
                if road_option in [ROAD_OPTIONS.STRAIGHT, ROAD_OPTIONS.LEFT, ROAD_OPTIONS.RIGHT]:
                    non_follow_event_happened = True
                else:
                    non_follow_event_happened = False

            #print("last waypoint generation", last_waypoint.transform.location.x, last_waypoint.transform.location.y)
            #print("during waypoint generation", next_waypoint.transform.location.x, next_waypoint.transform.location.y)

            if non_follow_event_happened:
                # change past
                NUM_PAST_WP = 5
                for i in range(min(NUM_PAST_WP, len(self._waypoints_queue))):
                    index = -(i+1)
                    wp, option, diff_angle0 = self._waypoints_queue[index]
                    self._waypoints_queue[index] = (wp, road_option, diff_angle0)
                    '''
                    if self._waypoints_queue[index][1] not in [ROAD_OPTIONS.STRAIGHT, ROAD_OPTIONS.LEFT, ROAD_OPTIONS.RIGHT]:
                        self._waypoints_queue[index][1] = road_option
                    else:
                        break
                    '''

                NUM_FUTURE_WP = 8
                self.count_down = NUM_FUTURE_WP
                self.count_down_value = road_option

            self._waypoints_queue.append((next_waypoint, road_option, diff_angle))

    def run_step(self, debug=False):
        """
        Execute one step of local planning which involves running the longitudinal and lateral PID controllers to
        follow the waypoints trajectory.

        :param debug: boolean flag to activate waypoints debugging
        :return:
        """

        # not enough waypoints in the horizon? => add more!
        if len(self._waypoints_queue) < int(self._waypoints_queue.maxlen * 0.5):
            self._compute_next_waypoints(k=10)

        # current vehicle waypoint
        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        #cw = self._current_waypoint
        #print("current waypoint is ", cw.transform.location.x, cw.transform.location.y)

        # target waypoint
        self._target_waypoint, self._current_road_option, diff_angle = self._waypoints_queue[0]
        #tw = self._target_waypoint
        #print("target waypoint is ", tw.transform.location.x, tw.transform.location.y)

        global_vars.set(diff_angle)
        #print("------------------------------------------------------------------->diff angle is ", diff_angle)
        # move using PID controllers
        recent_5_waypoints = [self._waypoints_queue[i][0] for i in range(5)]
        control, diff, adjusted_waypoint = self._vehicle_controller.run_step(self._target_speed, recent_5_waypoints)

        # purge the queue of obsolete waypoints
        vehicle_transform = self._vehicle.get_transform()
        max_index = -1

        # Fix a potential bug: the pattern might be False True False, but never True again
        # when we observe a second True, break the loop
        encountered_first_true = False
        encountered_false_after_true = False
        encountered_second_true = False
        for i, (waypoint, road_option, diff_angle) in enumerate(self._waypoints_queue):
            if distance_vehicle(waypoint, vehicle_transform) < self._min_distance:
                encountered_first_true = True
                if encountered_false_after_true:
                    encountered_second_true = True
                    break
                max_index = i
            else:
                if encountered_first_true:
                    encountered_false_after_true = True

        if max_index >= 0:
            for i in range(max_index + 1):
                self._waypoints_queue.popleft()

        if debug:
            # the original target waypoint
            draw_waypoints(self._vehicle.get_world(), [self._target_waypoint], z=0.5)

            # the all future waypoint
            wp_queue = []
            for wp in self._waypoints_queue:
                w = wp[0]
                wp_queue.append([w.transform.location.x, w.transform.location.y])
            draw_waypoints_norotation(self._vehicle.get_world(), wp_queue, z=0.5, color=carla.Color(r=0,g=0,b=255))

            # the adjusted waypint
            draw_waypoints_norotation(self._vehicle.get_world(), [adjusted_waypoint], z=0.5)

        map_option_to_numeric = {
            ROAD_OPTIONS.LANEFOLLOW: 2.0,
            ROAD_OPTIONS.LEFT: 3.0,
            ROAD_OPTIONS.RIGHT: 4.0,
            ROAD_OPTIONS.STRAIGHT: 5.0,
            ROAD_OPTIONS.VOID: 2.0
        }

        return control, map_option_to_numeric[self._current_road_option]

def retrieve_options(list_waypoints, current_waypoint):
    """
    Compute the type of connection between the current active waypoint and the multiple waypoints present in
    list_waypoints. The result is encoded as a list of ROAD_OPTIONS enums.

    :param list_waypoints: list with the possible target waypoints in case of multiple options
    :param current_waypoint: current active waypoint
    :return: list of ROAD_OPTIONS enums representing the type of connection from the active waypoint to each
             candidate in list_waypoints
    """
    options = []
    for next_waypoint in list_waypoints:
        # this is needed because something we are linking to
        # the beggining of an intersection, therefore the
        # variation in angle is small
        next_next_waypoint = next_waypoint.next(3.0)[0]
        link = compute_connection(current_waypoint, next_next_waypoint)
        options.append(link)

    return options

def compute_connection(current_waypoint, next_waypoint):
    """
    Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
    (next_waypoint).

    :param current_waypoint: active waypoint
    :param next_waypoint: target waypoint
    :return: the type of topological connection encoded as a ROAD_OPTION enum:
             ROAD_OPTIONS.STRAIGHT
             ROAD_OPTIONS.LEFT
             ROAD_OPTIONS.RIGHT
    """
    n_ = next_waypoint.transform.rotation.yaw
    n_ = n_ % 360.0

    c_ = current_waypoint.transform.rotation.yaw
    c_ = c_ % 360.0

    diff_angle = (n_ - c_) % 180.0
    if diff_angle < 10.0 or diff_angle > 170.0:
        return ROAD_OPTIONS.STRAIGHT, diff_angle
    elif diff_angle > 90.0:
        print("left because angle is ", diff_angle)
        return ROAD_OPTIONS.LEFT, diff_angle
    else:
        print("right because angle is ", diff_angle)
        return ROAD_OPTIONS.RIGHT, diff_angle
