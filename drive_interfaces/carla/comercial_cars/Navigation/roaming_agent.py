#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import carla, math
from Navigation.local_planner import *
from Tools.misc import *
import numpy as np

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

    DISTANCE_LIGHT = 10  # m
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

        '''
        self._last_red_light_id = None
        self._list_traffic_lights = []
        all_actors = self._world.get_actors()
        for _actor in all_actors:
            if 'traffic_light' in _actor.type_id:
                center, area = self.get_traffic_light_area(_actor)
                waypoints = []
                for pt in area:
                    waypoints.append(self._map.get_waypoint(pt))
                self._list_traffic_lights.append((_actor, center, area, waypoints))
        '''

    def rotate_point(self, pt, angle):
        x_ = math.cos(math.radians(angle)) * pt.x - math.sin(math.radians(angle)) * pt.y
        y_ = math.sin(math.radians(angle)) * pt.x - math.cos(math.radians(angle)) * pt.y
        return carla.Vector3D(x_, y_, pt.z)

    def get_traffic_light_area(self, tl):
        base_transform = tl.get_transform()
        base_rot = base_transform.rotation.yaw

        area_loc = base_transform.transform(tl.trigger_volume.location)

        wpx = self._map.get_waypoint(area_loc)
        while not wpx.is_intersection:
            next = wpx.next(1.0)[0]
            if next:
                wpx = next
            else:
                break
        wpx_location = wpx.transform.location
        area_ext = tl.trigger_volume.extent

        area = []
        # why the 0.9 you may ask?... because the triggerboxes are set manually and sometimes they
        # cross to adjacent lanes by accident
        x_values = np.arange(-area_ext.x * 0.9, area_ext.x * 0.9, 1.0)
        for x in x_values:
            pt = self.rotate_point(carla.Vector3D(x, 0, area_ext.z), base_rot)
            area.append(wpx_location + carla.Location(x=pt.x, y=pt.y))

        return area_loc, area

    def _is_light_red(self):
        """
        Method to check if there is a red light affecting us. This version of
        the method is compatible with both European and US style traffic lights.
        :param:
        :return: a tuple given by (bool_flag, traffic_light), where
                 - bool_flag is True if there is a traffic light in RED
                   affecting us and False otherwise
                 - traffic_light is the object itself or None if there is no
                   red traffic light affecting us
        """
        location = self._vehicle.get_transform().location
        if location is None:
            return (False, None)

        ego_waypoint = self._map.get_waypoint(location)

        for traffic_light, center, area, waypoints in self._list_traffic_lights:
            # logic
            center_loc = carla.Location(center)
            if self._last_red_light_id and self._last_red_light_id == traffic_light.id:
                continue
            if center_loc.distance(location) > self.DISTANCE_LIGHT:
                continue
            if traffic_light.state != carla.TrafficLightState.Red:
                continue

            for wp in waypoints:
                if ego_waypoint.road_id == wp.road_id and ego_waypoint.lane_id == wp.lane_id:
                    # this light is red and is affecting our lane!
                    # is the vehicle traversing the stop line?
                    self._last_red_light_id = traffic_light.id
                    return (True, traffic_light)

        return (False, None)

    def run_step(self, debug=False):
        """
        Execute one step of navigation.
        :return:
        """

        # is there an obstacle in front of us?
        hazard_detected = False
        current_location = self._vehicle.get_location()
        vehicle_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        #print("current position", current_location.x, current_location.y,
        #      "vehicle waypoint", vehicle_waypoint.transform.location.x, vehicle_waypoint.transform.location.y)

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

        '''
        light_state, traffic_light = self._is_light_red()
        if light_state:
            if debug:
                print('=== RED LIGHT AHEAD [{}])'.format(traffic_light.id))

            self._state = AGENT_STATE.BLOCKED_RED_LIGHT
            hazard_detected = True
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
            control, command = self._local_planner.run_step(debug)
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

