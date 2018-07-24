import sys
sys.path.append('drive_interfaces')
sys.path.append('drive_interfaces/carla')
sys.path.append('drive_interfaces/carla/carla_client')
sys.path.append('drive_interfaces/carla/comercial_cars')
sys.path.append('drive_interfaces/carla/carla_client/testing')
sys.path.append('test_interfaces')
sys.path.append('utils')
sys.path.append('dataset_manipulation')
sys.path.append('configuration')
sys.path.append('structures')
sys.path.append('evaluation')

import time, pygame, traceback, configparser, datetime
from noiser import Noiser
from screen_manager import ScreenManager

from drawing_tools import *
from extra import *
from common_util import preprocess_image

pygame.init()
clock = pygame.time.Clock()

def get_camera_dict(ini_file):
    config = configparser.ConfigParser()
    config.read(ini_file)
    cameras = config['CARLA/SceneCapture']['Cameras']
    camera_dict = {}
    cameras = cameras.split(',')
    print(cameras)
    for i in range(len(cameras)):
        angle = config['CARLA/SceneCapture/' + cameras[i]]['CameraRotationYaw']
        camera_dict.update({i: (cameras[i], angle)})

    return camera_dict


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
    folder_name = str(datetime.datetime.today().year) + \
                  str(datetime.datetime.today().month) + \
                  str(datetime.datetime.today().day)
    if drivers_name is not None:
        folder_name += '_' + drivers_name
    folder_name += '_' + str(get_latest_file_number(drive_config.path, folder_name))

    camera_dict = get_camera_dict(drive_config.carla_config)
    print(" Camera Dict ", camera_dict)
    recorder = Recorder(drive_config.path + folder_name + '/',
                        drive_config.resolution,
                        image_cut=drive_config.image_cut,
                        camera_dict=camera_dict)

    return driver, recorder


def drive(experiment_name, drive_config, name=None, memory_use=1.0):
    driver, recorder = get_instance(drive_config, experiment_name, name, memory_use)
    noiser = Noiser(drive_config.noise)

    print('before starting')
    driver.start()
    if drive_config.show_screen:
        screen_manager = ScreenManager()
        screen_manager.start_screen(drive_config.resolution,
                                    drive_config.aspect_ratio,
                                    drive_config.scale_factor)  # [800,600]

    direction = 2
    iteration = 0
    try:
        while direction != -1:
            capture_time = time.time()
            # get the sensory data
            measurements, direction = driver.get_sensor_data()  # Later it would return more image like [rewards,images,segmentation]
            # decide whether need recording
            recording = driver.get_recording()
            # reset the environment if needed
            driver.get_reset()

            # compute the actions based on the image and the speed
            speed = measurements.player_measurements.forward_speed
            actions = driver.compute_action([measurements['BGRA'][drive_config.middle_camera],
                                             measurements['Labels'][drive_config.middle_camera]],
                                            speed)  # measurements.speed
            action_noisy, drifting_time, will_drift = noiser.compute_noise(actions, speed)

            if recording:
                recorder.record(measurements, actions, action_noisy, direction, driver.get_waypoints())

            if drive_config.show_screen:
                if drive_config.interface == "Carla":
                    print('fps', 1.0 / (time.time() - capture_time))
                    image = preprocess_image(measurements['BGRA'][drive_config.middle_camera],
                                             drive_config.image_cut,
                                             None)
                    image.setflags(write=1)
                    screen_manager.plot_camera_steer(image, actions.steer, [0, 0])
                else:
                    raise ValueError("Not supported interface")

            if drive_config.type_of_driver == "Machine" and drive_config.show_screen and drive_config.plot_vbp:
                image_vbp = driver.compute_perception_activations(image, speed)
                screen_manager.plot_camera(image_vbp, [1, 0])

            iteration += 1
            driver.act(action_noisy)

    except:
        traceback.print_exc()

    finally:
        pygame.quit()
