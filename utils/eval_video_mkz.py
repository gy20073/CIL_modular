import os, sys, inspect, glob, h5py, cv2, copy, pickle
import numpy as np
from common_util import plot_waypoints_on_image
from subprocess import call
from PIL import Image, ImageDraw, ImageFont

def get_file_real_path():
    abspath = os.path.abspath(inspect.getfile(inspect.currentframe()))
    return os.path.realpath(abspath)

def get_driver_config():
    driver_conf = lambda: None  # an object that could add attributes dynamically
    driver_conf.image_cut = [0, 100000]
    driver_conf.host = None
    driver_conf.port = None
    driver_conf.use_planner = False  # fixed
    driver_conf.carla_config = None  # This is not used by CarlaMachine but it's required
    return driver_conf

# begin the configs
exp_id = "mm45_v4_base_newseg_noiser_TL_lane_structure02_goodsteer_waypoint_zoom_stdnorm_v5"
short_id = "v5"
use_left_right = False
video_path = "/scratch/yang/aws_data/mkz/mkz_large_fov/output_0_768.avi"
gpu = [0, 5]
direction_command = 2.0
speed_constant_kmh = 15.0

# end of the config
# The color encoding is: blue predicted, green ground truth, red approximated ground truth
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu[0])

driving_model_code_path = os.path.join(os.path.dirname(get_file_real_path()), "../")
os.chdir(driving_model_code_path)
sys.path.append("drive_interfaces/carla/comercial_cars")
from carla_machine import *

driving_model = CarlaMachine("0", exp_id, get_driver_config(), 0.1,
                             gpu_perception=gpu,
                             perception_paths="path_jormungandr_newseg",
                             batch_size=3 if use_left_right else 1)

def loop_over_video(path, func, temp_down_factor=1, batch_size=1, output_name="output.avi"):
    # from a video, use cv2 to read each frame

    # reading from a video
    cap = cv2.VideoCapture(path)

    i = 0
    batch_frames = []
    video_init = False
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        if i % temp_down_factor:
            i += 1
            continue
        print(i)
        batch_frames.append(frame)
        if len(batch_frames) == batch_size:
            # frame is the one
            print("calling loop function...")
            frame_seq = func(batch_frames)
            print("calling loop function finished")
            if not video_init:
                fourcc = cv2.VideoWriter_fourcc(*'DIVX')
                video = cv2.VideoWriter(output_name, fourcc, 30 // temp_down_factor,
                                        (frame_seq[0].shape[1], frame_seq[0].shape[0]))
                print("in test_video.loop_over_video, loop function output size:", frame_seq[0].shape)
                video_init = True
            for frame in frame_seq:
                video.write(frame)
            batch_frames = []
        i += 1

    cap.release()
    video.release()

wps = []
def callback(frames):
    frame = frames[0]
    sensors = [frame[:,:,::-1]]
    waypoints, to_be_visualized = driving_model.compute_action(sensors, speed_constant_kmh,
                                                                                  direction_command,
                                                                                  save_image_to_disk=False,
                                                                                  return_vis=True,
                                                                                  return_extra=False)
    wps.append(waypoints)
    path = video_path+".pkl"
    with open(path, "wb") as file:
        pickle.dump(wps, file)

    return [to_be_visualized]

loop_over_video(video_path,
                callback,
                temp_down_factor=1,
                batch_size=1,
                output_name=video_path+"."+short_id+".mp4")
