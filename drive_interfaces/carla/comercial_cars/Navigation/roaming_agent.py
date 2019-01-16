#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import carla
from Navigation.local_planner import *
from Tools.misc import *

class AGENT_STATE(Enum):
    """
    AGENT_STATE represents the possible states of a roaming agent
    """
    NAVIGATING = 1
    BLOCKED_BY_VEHICLE = 2
    BLOCKED_RED_LIGHT = 3

class RoamingAgent(object):
    """
    RoamingAgent implements a basic agent that navigates scenes making random choices when facing an intersection.
    This agent respects traffic lights and other vehicles.
    """
    def __init__(self, vehicle):
        """

        :param vehicle: actor to apply to local planner logic onto
        """
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._map = self._vehicle.get_world().get_map()
        self._proximity_threshold = 10.0 # meters
        self._state = AGENT_STATE.NAVIGATING
        self._local_planner = LocalPlanner(self._vehicle)

        self._last_command = 2.0

    def run_step(self, debug=False):
        """
        Execute one step of navigation.
        :return:
        """

        # is there an obstacle in front of us?
        hazard_detected = False
        current_location = self._vehicle.get_location()
        vehicle_waypoint = self._map.get_waypoint(self._vehicle.get_location())

        # retrieve relevant elements for safe navigation, i.e.: traffic lights and other vehicles
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")

        # TODO: disable the vehicle related hazard detection functionality, because we can not handle parking and moving vehicle at the same time
        '''
        # check possible obstacles
        for object in vehicle_list:
            # do not account for the ego vehicle
            if object.id == self._vehicle.id:
                continue
            # if the object is not in our lane it's not an obstacle
            object_waypoint = self._map.get_waypoint(object.get_location())
            if object_waypoint.road_id != vehicle_waypoint.road_id or object_waypoint.lane_id != vehicle_waypoint.lane_id:
                continue

            loc = object.get_location()
            if is_within_distance_ahead(loc, current_location, self._vehicle.get_transform().rotation.yaw,
                                          self._proximity_threshold):
                if debug:
                    print('!!! HAZARD [{}] ==> (x={}, y={})'.format(object.id, loc.x, loc.y))
                self._state = AGENT_STATE.BLOCKED_BY_VEHICLE
                hazard_detected = True
                break
        '''

        # check for the state of the traffic lights
        for object in lights_list:
            object_waypoint = self._map.get_waypoint(object.get_location())
            if object_waypoint.road_id != vehicle_waypoint.road_id or object_waypoint.lane_id != vehicle_waypoint.lane_id:
                continue

            loc = object.get_location()
            if is_within_distance_ahead(loc, current_location, self._vehicle.get_transform().rotation.yaw,
                                          self._proximity_threshold):
                if object.state == carla.libcarla.TrafficLightState.Red:
                    if debug:
                        print('=== RED LIGHT AHEAD [{}] ==> (x={}, y={})'.format(object.id, loc.x, loc.y))
                    self._state = AGENT_STATE.BLOCKED_RED_LIGHT
                    hazard_detected = True
                    break

        if hazard_detected:
            control = self.emergency_stop()
            command = self._last_command
        else:
            self._state = AGENT_STATE.NAVIGATING
            # standard local planner behavior
            control, command = self._local_planner.run_step()
            self._last_command = command

        return control, command

    def emergency_stop(self):
        """
        Send an emergency stop command to the vehicle
        :return:
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = False

        return control

