import sys
sys.path.append('../drive_interfaces/carla/carla_client_090/carla-0.9.1-py2.7-linux-x86_64.egg')

import carla, cv2, threading, copy
import numpy as np


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

        data_buffer_lock.acquire()
        self._obj._data_buffers[self._tag] = array
        data_buffer_lock.release()


class Carla090Eval():
    def __init__(self,
                 host='localhost',
                 port=2000,
                 vehicle_pos=carla.Transform(location=carla.Location(x=101.5, y=-69.0, z=3.0),
                                             rotation=carla.Rotation(roll=0, yaw=0, pitch=-69.4439547804))):
        # first create the carla evaluation environment
        self._client = carla.Client(host, port)
        self._client.set_timeout(2.0)
        self._world = self._client.get_world()

        # spawn a vehicle
        blueprints = self._world.get_blueprint_library().filter('vehicle')
        vechile_blueprint = [e for i, e in enumerate(blueprints) if e.id == 'vehicle.lincoln.mkz2017'][0]
        self._vehicle = self._world.try_spawn_actor(vechile_blueprint,vehicle_pos)

        # create all cameras
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

        cam_blueprint = self._world.get_blueprint_library().find('sensor.camera.rgb')
        cam_blueprint.set_attribute('image_size_x', str(WINDOW_WIDTH))
        cam_blueprint.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        cam_blueprint.set_attribute('fov', str(CAMERA_FOV))

        self._camera_center = self._world.spawn_actor(cam_blueprint, CAMERA_CENTER_TRANSFORM, attach_to=self._vehicle)
        self._camera_left = self._world.spawn_actor(cam_blueprint, CAMERA_LEFT_TRANSFORM, attach_to=self._vehicle)
        self._camera_right = self._world.spawn_actor(cam_blueprint, CAMERA_RIGHT_TRANSFORM, attach_to=self._vehicle)

        self._data_buffers = {}
        self._camera_center.listen(CallBack('CameraMiddle', self))
        self._camera_left.listen(CallBack('CameraLeft', self))
        self._camera_right.listen(CallBack('CameraRight', self))


        # load the trained model
        self.load_trained_model()

    def load_trained_model(self):
        pass

    def compute_control(self, image_dict):
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False
        control.reverse = False
        return control

    def show_image_dict_on_screen(self, image_dict):
        image = np.concatenate((image_dict['CameraLeft'], image_dict['CameraMiddle'], image_dict['CameraRight']), axis=1)
        cv2.imshow('viz', image[::2,::2,::-1])

    def save_to_disk(self):
        pass

    def run(self):
        while True:
            self._world.wait_for_tick(10.0)
            data_buffer_lock.acquire()
            image_dict = copy.deepcopy(self._data_buffers)
            data_buffer_lock.release()
            if len(image_dict) < 3:
                print("image dict has len ", len(image_dict), " elements, continuing")
                continue

            self.show_image_dict_on_screen(image_dict)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            self.save_to_disk()

            control = self.compute_control(image_dict)
            self._vehicle.apply_control(control)

eval_instance = Carla090Eval()
eval_instance.run()
