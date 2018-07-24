import sys, pygame, scipy, cv2, random
import tensorflow as tf
from pygame.locals import *
import numpy as np
from PIL import ImageDraw, Image, ImageFont

sys.path.append('../train')
sldist = lambda c1, c2: math.sqrt((c2[0] - c1[0])**2 + (c2[1] - c1[1])**2)
# carla related import
from carla.agent.agent import Agent
from carla.client import VehicleControl
from carla.client import make_carla_client
from carla import image_converter

from codification import *
from training_manager import TrainManager
import machine_output_functions
from driver import Driver
from drawing_tools import *
from common_util import restore_session, preprocess_image
slim = tf.contrib.slim

def load_system(config):
    config.batch_size = 1
    config.is_training = False

    training_manager = TrainManager(config, None)
    if hasattr(config, 'seg_network_erfnet_one_hot'):
        training_manager.build_seg_network_erfnet_one_hot()
        print("Bulding: seg_network_erfnet_one_hot")
    else:
        training_manager.build_network()
        print("Bulding: standard_network")

    return training_manager


class CarlaMachine(Agent, Driver):
    def __init__(self, gpu_number="0", experiment_name='None', driver_conf=None, memory_fraction=0.9):
        Driver.__init__(self)

        conf_module = __import__(experiment_name)
        self._config = conf_module.configInput()
        self._config.train_segmentation = False
        self._train_manager = load_system(conf_module.configTrain())

        config_gpu = tf.ConfigProto()
        config_gpu.gpu_options.visible_device_list = gpu_number
        config_gpu.gpu_options.per_process_gpu_memory_fraction = memory_fraction
        self._sess = tf.Session(config=config_gpu)

        self._sess.run(tf.global_variables_initializer())
        variables_to_restore = tf.global_variables()
        saver = tf.train.Saver(variables_to_restore)
        restore_session(self._sess, saver, self._config.models_path)

        self._control_function = getattr(machine_output_functions, self._train_manager._config.control_mode)
        self._image_cut = driver_conf.image_cut

        assert(driver_conf.use_planner == False)

        self._host = driver_conf.host
        self._port = driver_conf.port
        self._config_path = driver_conf.carla_config

        self._straight_button = False
        self._left_button = False
        self._right_button = False

        self.debug_i = 0
        self.temp_image_path = "./temp/"

    def start(self):
        self.carla = make_carla_client(self._host, self._port)
        self.positions = self.carla.loadConfigurationFile(self._config_path)
        self.carla.newEpisode(random.randint(0, len(self.positions)))
        self._target = random.randint(0, len(self.positions))

    def _get_direction_buttons(self):
        # with suppress_stdout():if keys[K_LEFT]:
        keys = pygame.key.get_pressed()

        if (keys[K_s]):
            self._left_button = False
            self._right_button = False
            self._straight_button = False

        if (keys[K_a]):
            self._left_button = True
            self._right_button = False
            self._straight_button = False

        if (keys[K_d]):
            self._right_button = True
            self._left_button = False
            self._straight_button = False

        if (keys[K_w]):
            self._straight_button = True
            self._left_button = False
            self._right_button = False

        return [self._left_button, self._right_button, self._straight_button]

    def compute_direction(self, pos, ori):  # This should have maybe some global position... GPS stuff
        # Button 3 has priority
        if 'Control' not in set(self._config.inputs_names):
            return None

        button_vec = self._get_direction_buttons()
        if sum(button_vec) == 0:  # Nothing
            return 2
        elif button_vec[0] == True:  # Left
            return 3
        elif button_vec[1] == True:  # Right
            return 4
        else:
            return 5

    def get_recording(self):
        return False

    def get_reset(self):
        return False

    # TODO: change to the agent interface
    def run_step(self, measurements, sensor_data, direction, target):
        sensors = []
        for name in self._config.sensor_names:
            if name == 'rgb':
                sensors.append(image_converter.to_bgra_array(sensor_data["CameraRGB"]))

        speed_KMH = measurements.player_measurements.forward_speed * 3.6
        control = self.compute_action(sensors, speed_KMH, direction)

        return control

    @staticmethod
    def write_text_on_image(image, string, fontsize=10):
        image = image.copy()
        image = np.uint8(image)
        j = Image.fromarray(image)
        draw = ImageDraw.Draw(j)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", fontsize)
        draw.text((0, 0), string, (255, 0, 0), font=font)

        return np.array(j)

    def save_image(self, sensor, direction):
        debug_path = self.temp_image_path + "/"
        txtdt = {2.0: "follow",
                 3.0: "left",
                 4.0: "right",
                 5.0: "straight",
                 0.0: "goal"}
        viz = self.write_text_on_image(sensor, txtdt[direction], 10)
        cv2.imwrite(debug_path +
                    str(self.debug_i).zfill(9) +
                    ".png", viz)
        self.debug_i += 1
        print("output image id is: ", self.debug_i)

    def compute_action(self, sensors, speed, direction=None):
        if direction == None:
            direction = self.compute_direction((0, 0, 0), (0, 0, 0))

        assert(len(self._config.sensor_names) == 1)
        assert(self._config.sensor_names[0] == 'rgb')
        image_input = preprocess_image(sensors[0], self._image_cut, self._config.sensors_size[0])
        self.save_image(image_input, direction)

        image_input = image_input.astype(np.float32)
        image_input = np.multiply(image_input, 1.0 / 255.0)

        if (self._train_manager._config.control_mode == 'single_branch_wp'):
            # Yang: use the waypoints to predict the steer, in theory PID controller, but in reality just P controller
            # TODO: ask, only the regression target is different, others are the same
            steer, acc, brake, wp1angle, wp2angle = \
                self._control_function(image_input, speed, direction,
                                       self._config, self._sess, self._train_manager)

            steer_pred = steer

            steer_gain = 0.8
            steer = steer_gain * wp1angle
            steer =min(max(steer, -1), 1)

            print(('Predicted Steering: ', steer_pred, ' Waypoint Steering: ', steer))
        else:
            steer, acc, brake = self._control_function(image_input, speed, direction,
                                                       self._config, self._sess, self._train_manager)

        if brake < 0.1 or acc > brake:
            brake = 0.0

        if speed > 35.0 and brake == 0.0:
            acc = 0.0

        control = VehicleControl()
        control.steer = steer
        control.throttle = acc
        control.brake = brake
        control.hand_brake = 0
        control.reverse = 0

        return control

    # The augmentation should be dependent on speed
    def get_sensor_data(self):
        measurements, _ = self.carla.read_data()
        direction = 2.0

        return measurements, direction

    def compute_perception_activations(self, sensor, speed):
        sensor = scipy.misc.imresize(sensor, [self._config.network_input_size[0], self._config.network_input_size[1]])
        image_input = sensor.astype(np.float32)
        image_input = np.multiply(image_input, 1.0 / 255.0)
        vbp_image = machine_output_functions.seg_viz(image_input, speed, self._config, self._sess, self._train_manager)

        return 0.4 * grayscale_colormap(np.squeeze(vbp_image), 'jet') + 0.6 * image_input  # inferno

    def act(self, action):
        self.carla.sendCommand(action)

    def stop(self):
        self.carla.stop()
