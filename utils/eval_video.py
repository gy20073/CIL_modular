import os, sys, inspect, glob, h5py, cv2, copy
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


def write_text_on_image(image, string, fontsize=10):
    image = image.copy()
    image = np.uint8(image)
    j = Image.fromarray(image)
    draw = ImageDraw.Draw(j)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", fontsize)
    draw.text((0, 0), string, (255, 0, 0), font=font)

    return np.array(j)

# begin the configs
exp_id = "mm45_v4_base_newseg_noiser_TL_lane_structure02_goodsteer_waypoint_zoom_cls"
use_left_right = False
h5path = "/data/yang/code/aws/scratch/carla_collect/steer103_v3_waypoint"
gpu = [0]
# end of the config

val_db_path = []
for valid in range(1, 15, 3):
    val_db_path += glob.glob(h5path+"/*WeatherId=" + str(valid).zfill(2) + "/data_*.h5")

driving_model_code_path = os.path.join(os.path.dirname(get_file_real_path()), "../")
os.chdir(driving_model_code_path)
sys.path.append("drive_interfaces/carla/comercial_cars")
from carla_machine import *

driving_model = CarlaMachine("0", exp_id, get_driver_config(), 0.1,
                             gpu_perception=gpu,
                             perception_paths="path_jormungandr_newseg",
                             batch_size=3 if use_left_right else 1)

centers = np.load(open("utils/cluster_centers.npy", "rb"))

for h5 in sorted(val_db_path):
    dirname, tail = os.path.split(h5)
    print(h5)
    f = h5py.File(h5, "r")
    targets = f["targets"]
    for i in range(200):
        print(i)
        vehicle_real_speed_kmh = targets[i, 10] * 3.6

        if use_left_right:
            sensor_names = ["CameraLeft", "CameraMiddle", "CameraRight"]
        else:
            sensor_names = ["CameraMiddle"]

        sensors = []
        for cam in sensor_names:
            img = cv2.imdecode(f[cam][i], 1)
            sensors.append(img)
        direction = targets[i, 24]

        waypoints, to_be_visualized, nrow, ncol, col_i = driving_model.compute_action(sensors, vehicle_real_speed_kmh, direction,
                                                save_image_to_disk=False, return_vis=True)

        # also plot the ground truth waypoints and the approximated waypoints
        image = to_be_visualized[:to_be_visualized.shape[0] // nrow,
                  col_i * to_be_visualized.shape[1] // ncol:(col_i + 1) * to_be_visualized.shape[1] // ncol, :]

        # then we have the waypoints stored
        imid = i
        size = int(f["targets"][imid, 99])
        flattend = f["targets"][imid, 35:(35 + size)]
        wp = np.reshape(flattend, (-1, 2))
        image = plot_waypoints_on_image(image, wp, 4, shift_ahead=2.46 - 0.7 + 2.0, rgb=(0, 255, 0))

        # also plot the cluster center corresponded
        ncluster = len(centers)
        cid = int(f["targets"][imid, 55])
        if cid < ncluster:
            wp = copy.deepcopy(centers[cid])
            wp *= f["targets"][imid, 56]
            wp = np.reshape(wp, (-1, 2))

            image = plot_waypoints_on_image(image, wp, 4, shift_ahead=2.46 - 0.7 + 2.0, rgb=(0, 0, 255))

        td = lambda fl: "{:.2f}".format(fl)
        font = int(np.ceil(15.0 / (576 / 2) * image.shape[0])) + 1
        image = write_text_on_image(image,
                                    "steer    :" + td(f["targets"][imid, 0]) + "\n" +
                                    "throttle :" + str(f["targets"][imid, 1]) + "\n" +
                                    "brake    :" + str(f["targets"][imid, 2]) + "\n" +
                                    "direction:" + str(f["targets"][imid, 24]) + "\n" +
                                    "speed    :" + td(f["targets"][imid, 10]) + "\n" +
                                    "ori      :" + td(f["targets"][imid, 21]) + " " + td(
                                        f["targets"][imid, 22]) + " " + td(f["targets"][imid, 23]) + "\n" +
                                    "wp1_angle:" + td(f["targets"][imid, 31]) + "\n",
                                    fontsize=font)

        to_be_visualized[:to_be_visualized.shape[0] // nrow,
        col_i * to_be_visualized.shape[1] // ncol:(col_i + 1) * to_be_visualized.shape[1] // ncol, :] = image

        # save it to the disk
        cv2.imwrite(dirname + "/" +
                    str(i).zfill(9) +
                    ".png", to_be_visualized[:,:,::-1])

    f.close()

    # make it into a video and delete the temp images
    out_name =  h5 + "_val.mp4"
    cmd = ["ffmpeg", "-y", "-i", dirname + "/%09d.png", "-c:v", "libx264", out_name]
    call(" ".join(cmd), shell=True)

    cmd = ["find", dirname, "-name", "00*png", "-print | xargs rm"]

    call(" ".join(cmd), shell=True)

