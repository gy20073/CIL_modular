# from scene_parameters import SceneParams

import argparse
import logging
import sys

sys.path.append('drive_interfaces/configuration')

sys.path.append('drive_interfaces')
sys.path.append('drive_interfaces/carla')
sys.path.append('drive_interfaces/carla/carla_client')

sys.path.append('drive_interfaces/carla/carla_client/planner')

sys.path.append('drive_interfaces/carla/carla_client/testing')

sys.path.append('test_interfaces')
sys.path.append('utils')
sys.path.append('dataset_manipulation')
sys.path.append('configuration')
sys.path.append('input')
sys.path.append('train')
sys.path.append('utils')
sys.path.append('input/spliter')
sys.path.append('structures')

from carla_machine import *

from carla.driving_benchmark import run_driving_benchmark
from carla.driving_benchmark.experiment_suites import CVPR2017


def parse_drive_arguments(args, driver_conf):
    # Carla Config
    if args.carla_config is not None:
        driver_conf.carla_config = args.carla_config

    if args.host is not None:
        driver_conf.host = args.host

    if args.port is not None:
        driver_conf.port = args.port

    if args.path is not None:
        driver_conf.path = args.path

    if args.driver is not None:
        driver_conf.type_of_driver = args.driver

    if args.resolution is not None:
        res_string = args.resolution.split(',')
        resolution = []
        resolution.append(int(res_string[0]))
        resolution.append(int(res_string[1]))
        driver_conf.resolution = resolution

    if args.city is not None:
        driver_conf.city_name = args.city

    if args.image_cut is not None:
        cut_string = args.image_cut.split(',')
        image_cut = []
        image_cut.append(int(cut_string[0]))
        image_cut.append(int(cut_string[1]))
        driver_conf.image_cut = image_cut

    return driver_conf

def main(host, port, city, summary_name, agent):
    #TODO: make an agent; define the camera in the testing env; change city name
    # debug Yang, after debug, change continue experiment to True
    experiment_suite = CVPR2017(city)
    run_driving_benchmark(agent, experiment_suite,
                          city_name=city,
                          log_name=summary_name,
                          continue_experiment=False,
                          host=host,
                          port=int(port),
                          save_images=True)


if (__name__ == '__main__'):

    parser = argparse.ArgumentParser(description='Chauffeur')

    parser.add_argument('-lg', '--log', help="activate the log file", action="store_true")
    parser.add_argument('-db', '--debug', help="put the log file to screen", action="store_true")

    # Train
    # TODO: some kind of dictionary to change the parameters
    parser.add_argument('-e', '--experiment-name',
                        help="The experiment name (NAME.py file should be in configuration folder, and the results will be saved to models/NAME)",
                        default="")

    parser.add_argument('-m', '--memory', default=1.0, help='The amount of memory this process is going to use')
    # Drive
    parser.add_argument('-cc', '--carla-config', help="Carla config file used for driving")
    parser.add_argument('-l', '--host', type=str, default='127.0.0.1', help='The IP where DeepGTAV is running')
    parser.add_argument('-p', '--port', default=8000, help='The port where DeepGTAV is running')
    parser.add_argument('-pt', '--path', type=str, default="/media/adas/012B4138528FF294/TestBranchNoCol2/",
                        help='Path to Store outputs')
    parser.add_argument('-res', '--resolution', default="800,600", help='If we are showing the screen of the player')
    parser.add_argument('--driver', default="Human", help='Select who is driving, a human or a machine')
    parser.add_argument('-s', '--summary', default="summary_number_1", help='summary')
    parser.add_argument('-cy', '--city', help='select the graph from the city being used')
    parser.add_argument('-imc', '--image_cut', help='Set the positions where the image is cut')

    args = parser.parse_args()
    if args.log or args.debug:
        LOG_FILENAME = 'log_runtests.log'
        logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
        if args.debug:  # set of functions to put the logging to screen
            root = logging.getLogger()
            root.setLevel(logging.DEBUG)
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            root.addHandler(ch)

    driver_conf_module = __import__("9cam_agent_carla_test_rc")
    driver_conf = driver_conf_module.configDrive()
    driver_conf.use_planner = True
    driver_conf = parse_drive_arguments(args, driver_conf)
    print (driver_conf)

    with  open(args.summary + '.stats.csv', 'w+') as f:
        f.write(args.experiment_name + '\n')
        conf_module = __import__(args.experiment_name)
        config = conf_module.configMain()
        with open(config.models_path + '/checkpoint', 'r') as content_file:
            f.write(content_file.read())

    # instance your controller here
    runnable = CarlaMachine("0", args.experiment_name, driver_conf, float(args.memory))

    main(args.host, args.port, args.city, args.summary, runnable)
