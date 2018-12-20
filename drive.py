

import sys, os
sys.path.append('drive_interfaces')
sys.path.append('drive_interfaces/carla')

sys.path.append("drive_interfaces/carla/comercial_cars/Navigation/")
import global_vars
global_vars.init()

__CARLA_VERSION__ = os.getenv('CARLA_VERSION', '0.8.X')
if __CARLA_VERSION__ == '0.8.X':
    sys.path.append('drive_interfaces/carla/carla_client')
    from carla import image_converter

else:
    sys.path.append('drive_interfaces/carla/carla_client_090')
    sys.path.append('drive_interfaces/carla/carla_client_090/carla-0.9.1-py2.7-linux-x86_64.egg')

sys.path.append('drive_interfaces/carla/comercial_cars')
sys.path.append('drive_interfaces/carla/carla_client/testing')
sys.path.append('test_interfaces')
sys.path.append('utils')
sys.path.append('dataset_manipulation')
sys.path.append('configuration')
sys.path.append('structures')
sys.path.append('evaluation')

import time, pygame, traceback, configparser, datetime, glob
from noiser import Noiser
from screen_manager import ScreenManager
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from drawing_tools import *
from extra import *
from common_util import preprocess_image

pygame.init()
clock = pygame.time.Clock()

# TODO: TURN this into A FACTORY CLASS
def get_instance(drive_config, experiment_name, drivers_name, memory_use):
    if drive_config.interface == "Carla":
        from carla_recorder import Recorder

        if drive_config.type_of_driver == "Human":
            from carla_human import CarlaHuman
            driver = CarlaHuman(drive_config)
        else:
            from carla_machine import CarlaMachine
            driver = CarlaMachine("0", experiment_name, drive_config, memory_use)
    else:
        print(" Not valid interface is set ")
        raise ValueError()

    # prepare a folder name
    if not drive_config.re_entry:
        folder_name = str(datetime.datetime.today().year) + \
                      str(datetime.datetime.today().month) + \
                      str(datetime.datetime.today().day)
    else:
        folder_name = ""
    if drivers_name is not None:
        if folder_name != "":
            folder_name = folder_name + "_"
        folder_name += drivers_name
    if not drive_config.re_entry:
        folder_name += '_' + str(get_latest_file_number(drive_config.path, folder_name))

    num_files_in_folder = len(glob.glob(drive_config.path + folder_name + '/*.h5'))
    print("currently, there are %d files in this folder" % (num_files_in_folder,))
    recorder = Recorder(drive_config.path + folder_name + '/',
                        drive_config.resolution,
                        current_file_number=num_files_in_folder,
                        image_cut=drive_config.image_cut)

    return driver, recorder, num_files_in_folder


def write_text_on_image(image, string, fontsize=10, position=(0,0)):
    image = image.copy()
    image = np.uint8(image)
    j = Image.fromarray(image)
    draw = ImageDraw.Draw(j)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", fontsize)
    #font = ImageFont.load_default()
    draw.text(position, string, (255, 0, 0), font=font)

    return np.array(j)


def drive(experiment_name, drive_config, name=None, memory_use=1.0):
    driver, recorder, num_files_in_folder = get_instance(drive_config, experiment_name, name, memory_use)

    extra_dict = {}
    if hasattr(drive_config, "noise_frequency"):
        extra_dict["frequency"] = drive_config.noise_frequency
    if hasattr(drive_config, "noise_intensity"):
        extra_dict["intensity"] = drive_config.noise_intensity
    if hasattr(drive_config, "min_noise_time_amount"):
        extra_dict["min_noise_time_amount"] = drive_config.min_noise_time_amount
    if hasattr(drive_config, "no_noise_decay_stage"):
        extra_dict["no_noise_decay_stage"] = drive_config.no_noise_decay_stage
    if hasattr(drive_config, "use_tick"):
        extra_dict["use_tick"] = drive_config.use_tick
    if hasattr(drive_config, "time_amount_multiplier"):
        extra_dict["time_amount_multiplier"] = drive_config.time_amount_multiplier
    if hasattr(drive_config, "noise_std"):
        extra_dict["noise_std"] = drive_config.noise_std
    if hasattr(drive_config, "no_time_offset"):
        extra_dict["no_time_offset"] = drive_config.no_time_offset

    print("extra dict is ", extra_dict)

    noiser = Noiser(drive_config.noise, **extra_dict)
    num_has_collected = num_files_in_folder * recorder._number_images_per_file  # 200 is num images per h5 file

    if drive_config.num_images_to_collect <= num_has_collected:
        print("closing recorder")
        recorder.close()
        return True

    try:
        print('before starting')
        driver.start()
        if drive_config.show_screen:
            gameDisplay = pygame.display.set_mode(drive_config.resolution,
                                                  pygame.HWSURFACE | pygame.DOUBLEBUF)
            '''
            screen_manager = ScreenManager()
            screen_manager.start_screen(drive_config.resolution,
                                        drive_config.aspect_ratio,
                                        drive_config.scale_factor)  # [800,600]
            '''

        direction = 2

        if drive_config.type_of_driver != "Human":
            print(drive_config.type_of_driver, "!!!!!!!!!!!!!!!!!")
            conf_module = __import__(experiment_name)
            _config = conf_module.configInput()


        while direction != -1 and drive_config.num_images_to_collect > num_has_collected:
            capture_time = time.time()
            # get the sensory data

            # TODO: update all this part

            measurements, sensor_data, direction = driver.get_sensor_data()  # Later it would return more image like [rewards,images,segmentation]
            # decide whether need recording
            recording = driver.get_recording()

            # reset the environment if needed
            need_drop_recent_measurement = driver.get_reset()

            speed_kmh = measurements.player_measurements.forward_speed * 3.6

            sensors = []
            if drive_config.type_of_driver != "Human":
                for name in _config.sensor_names:
                    sensors.append(image_converter.to_bgra_array(sensor_data[name]))

            # this only goes to carla_human, not carla_machine
            actions = driver.compute_action(sensors, speed_kmh)  # measurements.speed
            action_noisy = noiser.compute_noise(actions, speed_kmh)

            print('>>>>>> DIFF steering = {}'.format(action_noisy.steer - actions.steer))

            if recording:
                num_has_collected += 1
                recorder.record(measurements, sensor_data, actions, action_noisy, direction, driver.get_waypoints())

            if need_drop_recent_measurement:
                num_frames_dropped = recorder.remove_current_and_previous()
                print("has dropped ", num_frames_dropped, " frames")
                num_has_collected -= num_frames_dropped

            if drive_config.show_screen:
                if drive_config.interface == "Carla":
                    print('FPS: {}'.format(1.0 / (time.time() - capture_time)))

                    import copy
                    sensor_data = copy.deepcopy(sensor_data)

                    if __CARLA_VERSION__ == '0.8.X':
                        image = preprocess_image(image_converter.to_bgra_array(sensor_data['CameraMiddle']),
                                             drive_config.image_cut,
                                             None)
                    else:
                        image = sensor_data['CameraMiddle']

                    mapping = {2.0: "follow", 3.0: "left", 4.0: "right", 5.0: "straight"}
                    image = write_text_on_image(image, mapping[direction], 30, (0, image.shape[0]-80))
                    image = write_text_on_image(image, '{:03.2f}'.format(speed_kmh), 30, (150, image.shape[0]-80))

                    diff_angle_global = global_vars.get()
                    if diff_angle_global is None:
                        output = -1.0
                    else:
                        output = diff_angle_global

                    image = write_text_on_image(image, '{:03.2f}'.format(output), 30, (300, image.shape[0] - 80))

                    image = pygame.surfarray.make_surface(np.transpose(image, (1, 0, 2)))
                    gameDisplay.blit(image, (0, 0))
                    pygame.display.update()
                    # todo: display other necessary info, include but not limited to actions.steer
                else:
                    raise ValueError("Not supported interface")

            if drive_config.type_of_driver == "Machine" and drive_config.show_screen and drive_config.plot_vbp:
                image_vbp = driver.compute_perception_activations(image, speed_kmh)
                screen_manager.plot_camera(image_vbp, [1, 0])

            driver.act(action_noisy)

        print("before returning true")
        return True

    except:
        traceback.print_exc()
        return False
    finally:
        print("in finally")
        if drive_config.show_screen:
            pygame.quit()
        print("after quit and before del")
        driver.__del__()
        print("closing recorder")
        recorder.close()
        print("clean up done")
