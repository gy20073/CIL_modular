import sys
sys.path.append('../drive_interfaces/carla/carla_client_090/carla-0.9.1-py2.7-linux-x86_64.egg')
import carla, cv2, threading, copy, os, inspect, math
import numpy as np
from carla import VehicleControl as VehicleControl

def get_file_real_path():
    abspath = os.path.abspath(inspect.getfile(inspect.currentframe()))
    return os.path.realpath(abspath)

driving_model_code_path = os.path.join(os.path.dirname(get_file_real_path()), "../")
os.chdir(driving_model_code_path)
sys.path.append('drive_interfaces/carla/carla_client_090/carla-0.9.1-py2.7-linux-x86_64.egg')
sys.path.append("drive_interfaces/carla/comercial_cars")
from carla_machine import *


# some configs
vehicle_initial_pos = carla.Transform(location=carla.Location(x=101.5, y=-69.0, z=3.0),
                                             rotation=carla.Rotation(roll=0, yaw=0, pitch=-69.4439547804))
condition=2.0

exp_id="mm45_v4_wp2town3cam_parallel_control_2p3town_map_sensor_dropout_rfssim_moremap_simv2"
gpu=0
video_output_name="eval_output.avi"
# end of all configs

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



def get_driver_config():
    driver_conf = lambda: None  # an object that could add attributes dynamically
    driver_conf.image_cut = [0, 100000]
    driver_conf.host = None
    driver_conf.port = None
    driver_conf.use_planner = False  # fixed
    driver_conf.carla_config = None  # This is not used by CarlaMachine but it's required
    return driver_conf


class Carla090Eval():
    def __init__(self,
                 host='localhost',
                 port=2000,
                 vehicle_pos=carla.Transform(location=carla.Location(x=101.5, y=-69.0, z=3.0),
                                             rotation=carla.Rotation(roll=0, yaw=0, pitch=-69.4439547804)),
                 video_output_name="eval_output.avi",
                 exp_id="mm45_v4_wp2town3cam_parallel_control_2p3town_map_sensor_dropout_rfssim_moremap_simv2",
                 gpu = 0):
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

        self._video_init = False
        self.video_output_name = video_output_name

        # load the trained model
        self.load_trained_model(exp_id, gpu)

    def load_trained_model(self, exp_id, gpu):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


        self.driving_model = CarlaMachine("0", exp_id, get_driver_config(), 0.1,
                                     gpu_perception=[gpu],
                                     perception_paths="path_jormungandr_newseg",
                                     batch_size=3)

    def compute_control_default(self, image_dict):
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False
        control.reverse = False
        return control


    def estimate_speed(self):
        vel = self._vehicle.get_velocity()
        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

        # speed in m/s
        return speed

    def compute_control(self, image_dict, condition):
        sensors = [image_dict['CameraLeft'], image_dict['CameraMiddle'], image_dict['CameraRight']]
        speed_ms = self.estimate_speed()
        v_transform = self._vehicle.get_transform()
        pos = [v_transform.location.x, v_transform.location.y]
        ori = [0, 0, v_transform.rotation.yaw]

        control, to_be_visualized = self.driving_model.compute_action(sensors, speed_ms * 3.6,
                                                                   condition,
                                                                   save_image_to_disk=False,
                                                                   return_vis=True,
                                                                   return_extra=False,
                                                                   mapping_support={"town_id": "10", "pos": pos, "ori": ori})
        return control, to_be_visualized

    def show_image_dict_on_screen(self, image_dict):
        image = np.concatenate((image_dict['CameraLeft'], image_dict['CameraMiddle'], image_dict['CameraRight']), axis=1)
        cv2.imshow('viz', image[::2,::2,::-1])

    def save_to_disk(self, to_be_viz, down_factor=2):
        to_be_viz = to_be_viz[::down_factor, ::down_factor, :]
        
        if not self._video_init:
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            self.video = cv2.VideoWriter(self.video_output_name, fourcc, 20,
                                    (to_be_viz.shape[1], to_be_viz.shape[0]))
            print("in test_video.loop_over_video, loop function output size:", to_be_viz.shape)
            self._video_init = True

        self.video.write(to_be_viz)

    def run(self, condition):
        for i in range(300):
            self._world.wait_for_tick(10.0)
            data_buffer_lock.acquire()
            image_dict = copy.deepcopy(self._data_buffers)
            data_buffer_lock.release()
            if len(image_dict) < 3:
                print("image dict has len ", len(image_dict), " elements, continuing")
                continue

            action, to_be_viz = self.compute_control(image_dict, condition)

            if False:
                self.show_image_dict_on_screen(image_dict)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            self.save_to_disk(to_be_viz)
            vc = VehicleControl(float(action.throttle),
                                float(action.steer),
                                float(action.brake),
                                bool(action.hand_brake),
                                bool(action.reverse))
            self._vehicle.apply_control(vc)

            if i % 10 == 0:
                print(i)

eval_instance = Carla090Eval(vehicle_pos=vehicle_initial_pos,
                             video_output_name=video_output_name,
                             exp_id=exp_id,
                             gpu=gpu)
eval_instance.run(condition)
