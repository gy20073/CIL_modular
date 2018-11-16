import sys, os, inspect

curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

sys.path.append("../")
sys.path.append(os.path.join(curr_dir, "../../"))
from common import resize_images

def parse_drive_arguments(args, driver_conf, attributes):
    for attr in attributes:
        value = getattr(args, attr)
        if value is not None:
            setattr(driver_conf, attr, value)

    if hasattr(args, "port"):
        if args.port is not None:
            driver_conf.port = int(args.port)

    if hasattr(args, "resolution"):
        if args.resolution is not None:
            res_string = args.resolution.split(',')
            driver_conf.resolution = [int(res_string[0]), int(res_string[1])]

    if hasattr(args, "image_cut"):
        if args.image_cut is not None:
            cut_string = args.image_cut.split(',')
            driver_conf.image_cut = [int(cut_string[0]), int(cut_string[1])]

    return driver_conf

import os
import tensorflow as tf

def restore_session(sess, saver, models_path):
    if not os.path.exists(models_path):
        os.mkdir(models_path)
        os.mkdir(models_path + "/train/")
        os.mkdir(models_path + "/val/")

    ckpt = tf.train.get_checkpoint_state(models_path)
    if ckpt:
        print('Restoring from ', ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        ckpt = 0

    return ckpt

import scipy.misc

def preprocess_image(sensor, image_cut, sensor_size):
    sensor = sensor[image_cut[0]:image_cut[1], :, :3]
    sensor = sensor[:, :, ::-1]
    if sensor_size is not None and (sensor_size[0], sensor_size[1]) != (sensor.shape[0], sensor.shape[1]):
        #sensor = scipy.misc.imresize(sensor, [sensor_size[0], sensor_size[1]])
        sensor = cv2.resize(sensor, (sensor_size[1], sensor_size[0]))

    return sensor

import cv2
def split_camera_middle(sensor_data, sensor_names):
    id = sensor_names.index('CameraMiddle')
    rest_data = sensor_data[0:id] + sensor_data[(id+1):]
    middle = sensor_data[id]

    # now splitting the image into two smaller ones
    middle_shape = middle.shape
    middle = middle[middle.shape[0]//4: middle.shape[0]*3//4, :, :]
    left = middle[:, 0:middle.shape[1]//2, :]
    left = cv2.resize(left, (middle_shape[1], middle_shape[0]))
    right = middle[:, middle.shape[1]//2:, :]
    right = cv2.resize(right, (middle_shape[1], middle_shape[0]))
    rest_data += [left, right]

    return rest_data


def split_camera_middle_batch(sensor_data, sensor_names):
    id = sensor_names.index('CameraMiddle')
    rest_data = sensor_data[0:id] + sensor_data[(id+1):]
    middle = sensor_data[id]

    # now splitting the image into two smaller ones
    middle_shape = middle.shape # now shape is B H W C
    middle_shape = middle_shape[1:]
    middle = middle[:, middle_shape[0]//4: middle_shape[0]*3//4, :, :]
    left = middle[:, :, 0:middle_shape[1]//2, :]
    left = resize_images(left, (middle_shape[0], middle_shape[1]))
    right = middle[:, :, middle_shape[1]//2:, :]
    right = resize_images(right, (middle_shape[0], middle_shape[1]))
    rest_data += [left, right]

    return rest_data


def camera_middle_zoom(sensor_data, sensor_names, zoom_dict):
    out = {}
    for key in zoom_dict:
        id = sensor_names.index(key)
        # rest_data = sensor_data[0:id] + sensor_data[(id+1):]
        middle = sensor_data[id]

        if zoom_dict[key]:
            # now splitting the image into two smaller ones
            middle_shape = middle.shape
            middle = middle[middle_shape[0] // 4: middle_shape[0] * 3 // 4,
                     middle_shape[1] // 4: middle_shape[1] * 3 // 4, :]

        out[key] = middle

    ans = []
    for key in sensor_names:
        if key in out:
            ans.append(out[key])
        else:
            raise ValueError("zoom dict not complete")

    return ans

def camera_middle_zoom_batch(sensor_data, sensor_names, zoom_dict):
    out = {}
    for key in zoom_dict:
        id = sensor_names.index(key)
        # rest_data = sensor_data[0:id] + sensor_data[(id+1):]
        middle = sensor_data[id]

        if zoom_dict[key]:
            # now splitting the image into two smaller ones
            middle_shape = middle.shape # now shape is B H W C
            middle_shape = middle_shape[1:]
            middle = middle[:,
                            middle_shape[0] // 4: middle_shape[0] * 3 // 4,
                            middle_shape[1] // 4: middle_shape[1] * 3 // 4, :]
            middle = resize_images(middle, (middle_shape[0], middle_shape[1]))
        out[key] = middle

    ans = []
    for key in sensor_names:
        if key in out:
            ans.append(out[key])
        else:
            raise ValueError("zoom dict not complete")

    return ans


import numpy as np
import math
def plot_waypoints_on_image(image, wps, dot_size,
                            shift_ahead=2.46 - 0.7 + 2.0,
                            rgb=(255, 0, 0),
                            half_width_fov=math.radians(103.0)/2,
                            half_height_fov=math.radians(77.0)/2,
                            is_zoom=False):
    if is_zoom:
        half_width_fov = math.radians(58.0)/2
        half_height_fov = math.radians(41.0) / 2
        # todo change this value
        shift_ahead = 2.47 - 0.7 + 2.0 # 3.7
        shift_ahead = 4.6

    # assume image is bgr input
    imsize = image.shape
    wps = np.concatenate(([[0,0]], wps), axis=0)
    for i in range(wps.shape[0]):
        wp = wps[i]
        # the definition of the waypoint: wp[0] how far ahead (y), wp[1] left(-) right(+) (x)
        depth = wp[0] + shift_ahead
        horizontal = wp[1]
        vertical = -1.6
        h, v = point_to_2d(depth, horizontal, vertical, half_width_fov, half_height_fov)

        xoff = int((-v + 0.5) * imsize[0])
        yoff = int((h + 0.5) * imsize[1])
        image[xoff - dot_size: xoff + dot_size, yoff - dot_size: yoff + dot_size, 0] = rgb[2]
        image[xoff - dot_size: xoff + dot_size, yoff - dot_size: yoff + dot_size, 1] = rgb[1]
        image[xoff - dot_size: xoff + dot_size, yoff - dot_size: yoff + dot_size, 2] = rgb[0]

    return image


def point_to_2d(depth, horizontal, vertical, half_width_fov=math.radians(103.0)/2, half_height_fov = math.radians(77.0)/2):
    # the horizontal and vertical are both relative to the center, same as the output
    h = horizontal / depth * 0.5 / math.tan(half_width_fov)
    v = vertical   / depth * 0.5 / math.tan(half_height_fov)
    return h, v