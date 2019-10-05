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

# TODO begin the configs
exp_id = "mm45_v4_SqnoiseShoulder_rfsv6_withTL_fixTL"
short_id = "fixTL"
use_left_right = True
video_path = "/home/yang/data/aws_data/mkz/2019-06-18_18-06-23/video_enable.avi"
pickle_path = "/home/yang/data/aws_data/mkz/2019-06-18_18-06-23/video_enable.pkl"
gpu = [0]
offset = 0 # actually around 82 to 84
merge_follow_straight = True
# TODO end of the config

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

def loop_over_video(path, func, temp_down_factor=1, batch_size=1, output_name="output.avi", pickle_name=None):
    # from a video, use cv2 to read each frame

    # reading from a video
    cap = cv2.VideoCapture(path)
    with open(pickle_name, "rb") as fin:
        extra_info = pickle.load(fin)

    i = 0
    batch_frames = []
    video_init = False
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            print("breaking")
            break

        if i % temp_down_factor:
            i += 1
            continue
        print(i)
        batch_frames.append(frame)
        if len(batch_frames) == batch_size:
            # frame is the one
            print("calling loop function...")
            frame_seq = func(batch_frames, extra_info[i+offset])
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

def key_to_value(key):
    mapping = {"s": 5.0, "a": 3.0, "d": 4.0, "w": 2.0}
    if merge_follow_straight:
        mapping["s"]=2.0
    if key in mapping:
        return mapping[key]
    else:
        print("an unexpected key happened", key)
        return 2.0

wps = []
def callback(frames, extra_info):
    frame = frames[0]
    if use_left_right:
        # split into 3 cams
        H, W, C = frame.shape
        assert(W % 3 == 0)
        sensors = [frame[:, 0:W//3,:], frame[:, W//3:W*2//3,:], frame[:, W*2//3:, :]]
    else:
        sensors = [frame]
    if len(extra_info) == 2:
        speed_ms, condition = extra_info
        pos = None
        ori = None
    elif len(extra_info) == 4:
        # we require that the pos and ori should be directly input to the compute action function
        speed_ms, condition, pos, ori = extra_info
    else:
        # we require that the pos and ori should be directly input to the compute action function
        speed_ms, condition, pos, ori, gps = extra_info

    waypoints, to_be_visualized = driving_model.compute_action(sensors, speed_ms*3.6,
                                                               key_to_value(condition),
                                                               save_image_to_disk=False,
                                                               return_vis=True,
                                                               return_extra=False,
                                                               mapping_support={"town_id": "rfs", "pos": pos, "ori": ori})
    # TODO: temporally disable the waypoint saving mechanism
    #wps.append(waypoints)
    path = video_path+"."+short_id+".pkl"
    with open(path, "wb") as file:
        pickle.dump(wps, file)

    return [to_be_visualized[:,:,::-1]]

loop_over_video(video_path,
                callback,
                temp_down_factor=1,
                batch_size=1,
                output_name=video_path+"."+short_id+".mp4",
                pickle_name=pickle_path)
