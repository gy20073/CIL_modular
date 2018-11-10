import copy
import numpy as np
import threading
import weakref
import random
import os, sys
import time
from collections import namedtuple
import math as m

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

        if self._tag == "CameraMiddle":
            while self._obj._update_once == True:
                time.sleep(0.01)

        data_buffer_lock.acquire()
        self._obj._data_buffers[self._tag] = array
        data_buffer_lock.release()
        if self._tag == "CameraMiddle":
            self._obj._update_once = True

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


class ScenarioManager():
    def __init__(self, world, opts={}):
        self._world = world
        self._opts = opts
        self._spawn_points = list(self._world.get_map().get_spawn_points())
        self._actor_list = []
        self._hero_vehicle = None

        self._weather_list = ['ClearNoon', 'CloudyNoon', 'WetNoon', 'WetCloudyNoon',
                              'MidRainyNoon', 'HardRainNoon', 'SoftRainNoon', 'ClearSunset',
                              'CloudySunset', 'WetSunset', 'WetCloudySunset', 'MidRainSunset',
                              'HardRainSunset', 'SoftRainSunset']

        if 'weather' in opts:
            if opts['weather'] is None:
                # random weather
                self._current_weather = 0
                self._random_weather = True
            else:
                self._current_weather = opts['weather']
                self._random_weather = False
        else:
            # random weather
            self._current_weather = 0
            self._random_weather = True

        # sensors setup
        self._camera_center = None
        self._camera_left = None
        self._camera_right = None
        self._collision_sensor = None
        self._update_once = False
        self._data_buffers = dict()
        self._collision_events = []

        # TODO: Assign random position from file
        self.WINDOW_WIDTH = 768
        self.WINDOW_HEIGHT = 576
        self.CAMERA_FOV = 103.0

        CAMERA_CENTER_T = carla.Location(x=0.7, y=-0.0, z=1.60)
        CAMERA_LEFT_T = carla.Location(x=0.7, y=-0.4, z=1.60)
        CAMERA_RIGHT_T = carla.Location(x=0.7, y=0.4, z=1.60)

        CAMERA_CENTER_ROTATION = carla.Rotation(roll=0.0, pitch=0.0, yaw=0.0)
        CAMERA_LEFT_ROTATION = carla.Rotation(roll=0.0, pitch=0.0, yaw=-45.0)
        CAMERA_RIGHT_ROTATION = carla.Rotation(roll=0.0, pitch=0.0, yaw=45.0)

        self.CAMERA_CENTER_TRANSFORM = carla.Transform(location=CAMERA_CENTER_T, rotation=CAMERA_CENTER_ROTATION)
        self.CAMERA_LEFT_TRANSFORM = carla.Transform(location=CAMERA_LEFT_T, rotation=CAMERA_LEFT_ROTATION)
        self.CAMERA_RIGHT_TRANSFORM = carla.Transform(location=CAMERA_RIGHT_T, rotation=CAMERA_RIGHT_ROTATION)

        # autopilot setup
        if 'autopilot' in opts:
            self._autopilot = opts['autopilot']
        else:
            self._autopilot = False

        # initial positions
        if 'start_positions' in opts:
           self._start_positions = opts['start_positions']
        else:
            self._start_positions = np.zeros(())


        self.last_timestamp = lambda x: x
        self.last_timestamp.elapsed_seconds = 0.0
        self.last_timestamp.delta_seconds = 0.2

    def __del__(self):
        # destroy old actors
        print('destroying actors')
        self._clean_actors()
        print('destroying sensors')
        self._clean_sensors()
        print('done.')

    def _clean_actors(self):
        if self._hero_vehicle is not None:
            self._hero_vehicle.destroy()
            self._hero_vehicle = None
        for actor in self._actor_list:
            actor.destroy()
        self._actor_list = []

    def _clean_sensors(self):
        if self._camera_center is not None:
            self._camera_center.destroy()
            self._camera_center = None
        if self._camera_left is not None:
            self._camera_left.destroy()
            self._camera_left = None
        if self._camera_right is not None:
            self._camera_right.destroy()
            self._camera_right = None


    def hero_apply_control(self, control):
        self._hero_vehicle.apply_control(control)

    def get_sync_sensor_data(self):
        self.last_timestamp.elapsed_seconds += 0.2

        while self._update_once == False:
            time.sleep(0.01)

        data_buffer_lock.acquire()
        sensor_data = copy.deepcopy(self._data_buffers)
        data_buffer_lock.release()

        self._update_once = False

        collision_lock.acquire()
        collision_event = self._collision_events
        self._last_collided = len(collision_event) > 0
        self._collision_events = []
        collision_lock.release()

        if len(collision_event) > 0:
            print(collision_event)
        # TODO: make sure those events are actually valid
        if 'Static' in collision_event:
            collision_other = 1.0
        else:
            collision_other = 0.0
        if "Vehicles" in collision_event:
            collision_vehicles = 1.0
        else:
            collision_vehicles = 0.0
        if "Pedestrians" in collision_event:
            collision_pedestrians = 1.0
        else:
            collision_pedestrians = 0.0

        current_ms_offset = self.last_timestamp.elapsed_seconds * 1000

        # send all acquired data
        second_level = namedtuple('second_level', ['forward_speed', 'transform', 'collision_other', 'collision_pedestrians', 'collision_vehicles'])
        transform = namedtuple('transform', ['location', 'orientation'])
        loc = namedtuple('loc', ['x', 'y'])
        ori = namedtuple('ori', ['x', 'y', 'z'])
        Meas = namedtuple('Meas', ['player_measurements', 'game_timestamp'])

        v_transform = self._hero_vehicle.get_transform()
        vel = self._hero_vehicle.get_velocity()
        speed = m.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

        measurements = Meas(
                                second_level(speed,
                                             transform(loc(v_transform.location.x,
                                                           v_transform.location.y),
                                                       ori(v_transform.rotation.pitch,
                                                           v_transform.rotation.roll,
                                                           v_transform.rotation.yaw)),
                                             collision_other,
                                             collision_pedestrians,
                                             collision_vehicles),
                                current_ms_offset)

        return {'measurements':measurements, 'sensor_data':sensor_data}


    def _init_sensors(self, vehicle):
        cam_blueprint = self._world.get_blueprint_library().find('sensor.camera.rgb')

        cam_blueprint.set_attribute('image_size_x', str(self.WINDOW_WIDTH))
        cam_blueprint.set_attribute('image_size_y', str(self.WINDOW_HEIGHT))
        cam_blueprint.set_attribute('fov', str(self.CAMERA_FOV))

        if self._camera_center is not None:
            self._camera_center.destroy()
            self._camera_left.destroy()
            self._camera_right.destroy()
            self._camera_center = None

        if self._camera_center == None:
            self._camera_center = self._world.spawn_actor(cam_blueprint, self.CAMERA_CENTER_TRANSFORM,
                                                          attach_to=vehicle)
            self._camera_left = self._world.spawn_actor(cam_blueprint, self.CAMERA_LEFT_TRANSFORM, attach_to=vehicle)
            self._camera_right = self._world.spawn_actor(cam_blueprint, self.CAMERA_RIGHT_TRANSFORM, attach_to=vehicle)

            self._camera_center.listen(CallBack('CameraMiddle', self))
            self._camera_left.listen(CallBack('CameraLeft', self))
            self._camera_right.listen(CallBack('CameraRight', self))

        if self._collision_sensor is not None:
            self._collision_sensor.sensor.destroy()
        self._collision_sensor = CollisionSensor(vehicle, self)

    def _init_weathers(self):
        if self._random_weather:
            self._current_weather = random.randint(0, len(self._weather_list)-1)

        print('---- CUrrent weather = {}'.format(self._current_weather))
        weather_string =  self._weather_list[self._current_weather]
        weather = getattr(carla.WeatherParameters, weather_string)
        self._world.set_weather(weather)

    def _init_hero_vehicle_random(self):
        # create ego-vehicle
        blueprints = self._world.get_blueprint_library().filter('vehicle')
        vechile_blueprint = [e for i, e in enumerate(blueprints) if e.id == 'vehicle.lincoln.mkz2017'][0]

        if self._hero_vehicle == None or self._autopilot:
            if self._autopilot and self._hero_vehicle is not None:
                self._hero_vehicle.destroy()
                self._hero_vehicle = None

            while self._hero_vehicle == None:
                if self._autopilot:
                    # from designated points
                    START_POSITION = random.choice(self._spawn_points)
                else:
                    random_position = self._start_positions[np.random.randint(self._start_positions.shape[0]), :]
                    START_POSITION = carla.Transform(
                        carla.Location(x=random_position[0], y=random_position[1], z=random_position[2] + 1.0),
                        carla.Rotation(pitch=random_position[3], roll=random_position[4], yaw=random_position[5]))

                self._hero_vehicle = self._world.try_spawn_actor(vechile_blueprint, START_POSITION)
        else:
            if self._autopilot:
                # from designated points
                START_POSITION = random.choice(self._spawn_points)
            else:
                random_position = start_positions[np.random.randint(start_positions.shape[0]), :]
                START_POSITION = carla.Transform(
                    carla.Location(x=random_position[0], y=random_position[1], z=random_position[2] + 1.0),
                    carla.Rotation(pitch=random_position[3], roll=random_position[4], yaw=random_position[5]))

            self._hero_vehicle.set_transform(START_POSITION)

        if self._autopilot:
            self._hero_vehicle.set_autopilot()

        if self._collision_sensor is not None:
            self._collision_sensor.sensor.destroy()
        self._collision_sensor = CollisionSensor(self._hero_vehicle, self)

    def set_hero_vehicle(self, vehicle):
        self._hero_vehicle = vehicle

    def _try_spawn_random_vehicle_at(self, blueprints, transform):
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        vehicle = self._world.try_spawn_actor(blueprint, transform)
        if vehicle is not None:
            self._actor_list.append(vehicle)
            vehicle.set_autopilot()
            #print('spawned %r at %s' % (vehicle.type_id, transform.location))
            return True
        return False

    def set_scenario(self, id):
        if id == 'SCENARIO_RANDOM':
            self._set_scenario_random()
        elif id == 'SCENARIO_UNPRO_LEFT':
            self._set_scenario_unproleft()
        elif id == 'SCENARIO_INPRO_RIGHT':
            self._set_scenario_unproright()
        else:
            raise Error

    def _set_scenario_random(self):
        self._clean_actors()
        self._init_weathers()
        self._init_hero_vehicle_random()
        self._init_sensors(self._hero_vehicle)


        # add traffic
        blueprints_vehi = self._world.get_blueprint_library().filter('vehicle.*')
        blueprints_vehi = [x for x in blueprints_vehi if int(x.get_attribute('number_of_wheels')) == 4]
        blueprints_vehi = [x for x in blueprints_vehi if not x.id.endswith('isetta')]

        # @todo Needs to be converted to list to be shuffled.
        random.shuffle(self._spawn_points)

        print('found %d spawn points.' % len(self._spawn_points))

        count = 10
        for spawn_point in self._spawn_points:
            if self._try_spawn_random_vehicle_at(blueprints_vehi, spawn_point):
                count -= 1
            if count <= 0:
                break
        while count > 0:
            time.sleep(0.5)
            if self._try_spawn_random_vehicle_at(blueprints_vehi, random.choice(self._spawn_points)):
                count -= 1
                # end traffic addition!