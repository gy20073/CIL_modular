import sys
import traceback

sys.path.append('drive_interfaces')
sys.path.append('drive_interfaces/carla')
sys.path.append('drive_interfaces/carla/carla_client')
sys.path.append('drive_interfaces/carla/comercial_cars')

sys.path.append('drive_interfaces/carla/virtual_elektra')
sys.path.append('drive_interfaces/gta_interface')
sys.path.append('drive_interfaces/deeprc_interface')
sys.path.append('drive_interfaces/carla/carla_client/testing')
sys.path.append('test_interfaces')
sys.path.append('utils')
sys.path.append('dataset_manipulation')
sys.path.append('configuration')
sys.path.append('structures')
sys.path.append('evaluation')

import math, pygame
from noiser import Noiser
import configparser
import datetime

from screen_manager import ScreenManager

import numpy as np
import time

# from config import *
# from eConfig import *
from drawing_tools import *
from extra import *

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

    if hasattr(drive_config, 'carla_config'):
        camera_dict = get_camera_dict(drive_config.carla_config)
        print(" Camera Dict ")
        print(camera_dict)

    folder_name = str(datetime.datetime.today().year) + str(datetime.datetime.today().month) + str(
        datetime.datetime.today().day)

    if drivers_name is not None:
        folder_name += '_' + drivers_name
    folder_name += '_' + str(get_latest_file_number(drive_config.path, folder_name))

    if hasattr(drive_config, 'carla_config'):
        recorder = Recorder(drive_config.path + folder_name + '/', drive_config.resolution, \
                            image_cut=drive_config.image_cut, camera_dict=camera_dict, record_waypoints=True)
    else:  ##RC Car
        recorder = Recorder(drive_config.path + folder_name + '/', 88, 200, image_cut=drive_config.image_cut)

    return driver, recorder


def drive(experiment_name, drive_config, name=None, memory_use=1.0):
    # host,port,gpu_number,path,show_screen,resolution,noise_type,config_path,type_of_driver,experiment_name,city_name,game,drivers_name
    driver, recorder = get_instance(drive_config, experiment_name, name, memory_use)

    noiser = Noiser(drive_config.noise)

    print('before starting')
    driver.start()
    first_time = True
    if drive_config.show_screen:
        screen_manager = ScreenManager()
        screen_manager.start_screen(drive_config.resolution, drive_config.aspect_ratio,
                                    drive_config.scale_factor)  # [800,600]

    direction = 2

    iteration = 0
    try:
        while direction != -1:
            capture_time = time.time()

            if hasattr(drive_config, 'carla_config'):
                measurements, direction = driver.get_sensor_data()  # Later it would return more image like [rewards,images,segmentation]
            else:  ##RC Car
                measurements, images = driver.get_sensor_data()  # Later it would return more image like [rewards,images,segmentation]

            # Compute now the direction
            if drive_config.show_screen:
                for event in pygame.event.get():  # User did something
                    if event.type == pygame.QUIT:  # If user clicked close
                        done = True  # Flag that we are done so we exit this loop

            recording = driver.get_recording()
            driver.get_reset()
            if hasattr(drive_config, 'carla_config'):
                speed = measurements.player_measurements.forward_speed
                # actions = driver.compute_action(images.rgb[drive_config.middle_camera],measurements.forward_speed,\
                # driver.compute_direction((measurements.transform.location.x,measurements.transform.location.y,22),\
                # (measurements.transform.orientation.x,measurements.transform.orientation.y,measurements.transform.orientation.z))) #rewards.speed
                # actions = driver.compute_action(images.rgb[drive_config.middle_camera],measurements.forward_speed) #rewards.speed
                actions = driver.compute_action([measurements['BGRA'][drive_config.middle_camera],
                                                 measurements['Labels'][drive_config.middle_camera]],
                                                speed)  # measurements.speed
                action_noisy, drifting_time, will_drift = noiser.compute_noise(actions, speed)

            else:  ##RC Car
                actions = driver.compute_action(images[drive_config.middle_camera], 0)  # measurements.speed
                action_noisy, drifting_time, will_drift = noiser.compute_noise(actions[drive_config.middle_camera])

            # print actions
            if recording:
                if drive_config.interface == "DeepRC":

                    recorder.record(images, measurements, actions, action_noisy)
                else:
                    recorder.record(measurements, actions, action_noisy, direction, driver.get_waypoints())

            if drive_config.show_screen:
                if drive_config.interface == "Carla" or drive_config.interface == "VirtualElektra":
                    # for i in range(drive_config.aspect_ratio[0]*drive_config.aspect_ratio[1]):
                    print('fps', 1.0 / (time.time() - capture_time))

                    # print measurements['BGRA'][drive_config.middle_camera].shape
                    image = measurements['BGRA'][drive_config.middle_camera][
                            drive_config.image_cut[0]:drive_config.image_cut[1], :, :]
                    image = image[:, :, :3]
                    image = image[:, :, ::-1]
                    # print image.shape
                    image.setflags(write=1)
                    screen_manager.plot_camera_steer(image, actions.steer, [0, 0])
                    '''
                    #print measurements['Labels'][drive_config.middle_camera].shape
                    image = measurements['Labels'][drive_config.middle_camera]
                    image = image[:,:,2]*30
                    image = image[:,:,np.newaxis]
                    #print image.shape
                    #image = image[:, :, ::-1]
                    image.setflags(write=1)
                    screen_manager.plot_camera_steer(image,actions.steer,[1,0])
                    '''
                else:
                    print("Not supported interface")
                    pass

            if drive_config.type_of_driver == "Machine" and drive_config.show_screen and drive_config.plot_vbp:
                image_vbp = driver.compute_perception_activations(image, speed)
                screen_manager.plot_camera(image_vbp, [1, 0])

            iteration += 1
            driver.act(action_noisy)

    except:
        traceback.print_exc()

    finally:
        # driver.write_performance_file(path,folder_name,iteration)
        pygame.quit()
