import sys, os, cv2, math, inspect, copy
import numpy as np

def get_current_folder():
    return os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(get_current_folder() + "/../drive_interfaces/carla/carla_client_deprecated")

def rotate(im, angle_radian, output_size):
    m = im
    yaw = angle_radian

    M = cv2.getRotationMatrix2D((m.shape[1] // 2, m.shape[0] // 2), np.rad2deg(yaw), 1)
    dst = cv2.warpAffine(m, M, (m.shape[1], m.shape[0]))
    if len(dst.shape) == 2:
        dst = np.expand_dims(dst, axis=2)

    neighbour = output_size
    dst = dst[dst.shape[0] // 2 - neighbour: dst.shape[0] // 2 + neighbour,
              dst.shape[1] // 2 - neighbour: dst.shape[1] // 2 + neighbour, :]
    return dst


class mapping_helper:
    def __init__(self, output_physical_size_meter=30.0, output_height_pix=50, version="v1"):
        '''
        :param output_physical_size_meter:
        :param output_height_pix:
        :param version: v1 is the original road vs non road
                        v2 is the road mask + shoulder region
        '''

        self.version = version
        print("the version of mapping we are using is ", self.version)

        self.output_height_pix = output_height_pix
        # map path, meters per pixel
        infos = {"rfs": (get_current_folder()+"/data_lanes/human_marked5_"+version+".png", 0.272736441511),
                 "01" : (get_current_folder()+"/data_lanes/Town01Lanes_"+version+".png",   0.1643),
                 "02":  (get_current_folder()+"/data_lanes/Town02Lanes_"+version+".png",   0.1643),
                 "10": (get_current_folder() + "/data_lanes/rfs_sim_"+version+".png", 0.277045),
                 "11": (get_current_folder() + "/data_lanes/exptown_" + version + ".png", 0.145945),
                 "13": (get_current_folder() + "/data_lanes/exptown_noshoulder_" + version + ".png", 0.145945)} # rfs_sim
        self.maps = {}
        self.output_pixel_size = {}

        for key in infos:
            self.output_pixel_size[key] = int(output_physical_size_meter / infos[key][1])
            self.maps[key] = self.rgb_to_machine_format(self.load_image(infos[key][0]))
            # apply the padding
            padding = 3*self.output_pixel_size[key]
            self.maps[key] = np.pad(self.maps[key], ((padding, padding), (padding, padding), (0, 0)), 'constant')
        self.carla_map = None
        self.loc_to_pix = {"rfs": lambda loc: self.loc_to_pix_rfs(loc),
                           "01":  lambda loc: self.loc_to_pix_01_02(loc, "01"),
                           "02":  lambda loc: self.loc_to_pix_01_02(loc, "02"),
                           "10": lambda loc: self.loc_to_pix_rfs_sim(loc),
                           "11": lambda loc: self.loc_to_pix_exptown(loc),
                           "13": lambda loc: self.loc_to_pix_exptown(loc)} # rfs_sim
        self.output_physical_size_meter = output_physical_size_meter

    def loc_to_pix_exptown(self, loc):
        u = 6.848364717542121 * loc[1] + 1267.9073339940535
        v = -6.851075806443265 * loc[0] + 2504.8267451634106
        return [int(v), int(u)]

    def loc_to_pix_rfs_sim(self, loc):
        u = 3.6090651558073654 * loc[1] + 2500.541076487252
        v = -3.6103367739019054 * loc[0] + 2501.862578166202
        return [int(v), int(u)]

    def loc_to_pix_rfs(self, latlog):
        UL = [37.918355, -122.338461]
        LR = [37.911971, -122.328151]
        mapping = self.maps["rfs"]
        padding = 3 * self.output_pixel_size["rfs"]
        relx = int((latlog[0] - UL[0]) / (LR[0] - UL[0]) * (mapping.shape[0] - padding*2))
        rely = int((latlog[1] - UL[1]) / (LR[1] - UL[1]) * (mapping.shape[1] - padding*2))
        return [relx, rely]

    def loc_to_pix_01_02(self, location, town_name):
        if location[0] == "debug_direct_pixel":
            return location[1:]

        if self.carla_map is None:
            from carla.planner.map import CarlaMap
            self.carla_map = {"01": CarlaMap("Town01", 0.1643, 50.0),
                              "02": CarlaMap("Town02", 0.1643, 50.0)}
        cur = self.carla_map[town_name].convert_to_pixel([location[0], location[1], .22])
        cur = [int(cur[1]), int(cur[0])]
        return cur

    def load_image(self, path):
        im = cv2.imread(path)
        im = im[:, :, :3]
        im = im[:, :, ::-1]
        return im

    def rgb_to_machine_format(self, im):
        if self.version == "v1":
            # this returns a 3d image, H, W, Channel==1
            bin = np.sum(im, axis=2, keepdims=True)
            bin = (bin > 0)
            bin = bin.astype(np.uint8)
            return bin
        elif self.version == "v2" or self.version == "v3":
            im = (np.array(im) > 100)
            im = im.astype(np.uint8)
            return im

    def ori_to_yaw(self, ori, town_id):
        if town_id == "rfs":
            # assume we have already called the quaternion_to_yaw method, so
            return ori
        elif town_id == "01" or town_id == "02":
            yaw = np.arctan2(-ori[1], ori[0]) - np.pi / 2
            return -yaw
        elif town_id == "03" or town_id == "04" or town_id == "10" or town_id == "11" or town_id == "13": #rfs_sim
            ori0=np.cos(np.radians(ori[2]))
            ori1=np.sin(np.radians(ori[2]))
            yaw = np.arctan2(-ori1, ori0) - np.pi / 2

            return -yaw - np.pi/2

    def get_map(self, town_id, pos, ori):
        map = self.maps[town_id]
        func = self.loc_to_pix[town_id]
        pix = func(pos)
        padding = self.output_pixel_size[town_id] * 3
        pix = [pix[0]+padding, pix[1]+padding]

        crop_size = self.output_pixel_size[town_id] * 2
        if pix[0] - crop_size < 0 or pix[0] + crop_size > map.shape[0]:
            print("get map location 0, out of range", pix[0], map.shape)
            pix[0] = min(max(pix[0], crop_size), map.shape[0]-crop_size)
        if pix[1] - crop_size < 0 or pix[1] + crop_size > map.shape[1]:
            print("get map location 1, out of range", pix[1], map.shape)
            pix[1] = min(max(pix[1], crop_size), map.shape[1]-crop_size)

        cropped = map[pix[0]-crop_size: pix[0]+crop_size,
                      pix[1]-crop_size: pix[1]+crop_size, :]
        cropped = copy.deepcopy(cropped)
        yaw_radian = self.ori_to_yaw(ori, town_id)
        cropped = rotate(cropped, yaw_radian, self.output_pixel_size[town_id])
        # cut the lower 1/3
        dst = cropped[: cropped.shape[0] * 2 // 3, :, :]
        # resize
        # here dst has dim 3
        dst = cv2.resize(dst, (int(self.output_height_pix * 1.0 / dst.shape[0] * dst.shape[1]),
                               self.output_height_pix))
        # here dst has dimension 2, if the last channel is 1.

        if self.version == "v3":
            sz = 1
            h0 = dst.shape[0] * 3 // 2 // 2
            h1 = dst.shape[1] // 2
            dst[h0 - sz: h0 + sz, h1 - sz: h1 + sz, :] = np.array([0, 1, 0])

        return dst

    def map_to_debug_image(self, map):
        if self.version == "v1":
            im = np.stack((map, map, map), axis=2)
        else:
            im = map
        im = im * 255
        if self.version != "v3":
            sz = 2
            h0 = im.shape[0] * 3 // 2 // 2
            h1 = im.shape[1] // 2
            im[h0-sz: h0+sz, h1-sz: h1+sz, :] = np.array([38, 239, 232])
        return im


    def compute_dis_to_border(self, map):
        # return the distance to border(other part of the road), in meters
        # positive for inside the road, and negative for outside the road.
        if self.version=="v1":
            H, W = map.shape
        else: #elif self.version == "v2":
            H, W, C = map.shape
            map = map[:, :, 0] * map[:, :, 1] * map[:, :, 2]

        center = (H * 3 // 4, W // 2)
        cv = map[center[0], center[1]]
        mesh = np.meshgrid(range(H), range(W), indexing='ij')
        mesh = [mesh[0][map!=cv], mesh[1][map!=cv]]
        dists = np.square(mesh[0]-center[0]) + np.square(mesh[1]-center[1])
        min_dist = np.sqrt(np.min(dists)) / W * self.output_physical_size_meter
        if cv == False:
            min_dist = -min_dist
        return min_dist

    def is_on_road(self, map):
        # return true for onroad and false for not
        if self.version=="v1":
            H, W = map.shape
        else: #elif self.version == "v2":
            H, W, C = map.shape
            map = map[:, :, 0] * map[:, :, 1] * map[:, :, 2]

        center = (H * 3 // 4, W // 2)
        cv = map[center[0], center[1]]
        return cv == 1


# this should goes to Ros publishing the message
def quaternion_to_yaw(msg):
    q = msg.orientation
    #yaw = math.atan2(2 * (q.x * q.w + q.y * q.z), 1 - 2 * (q.z ** 2 + q.w ** 2))
    pitch = math.atan2(2*(q.x*q.y + q.z*q.w), 1-2*(q.y**2 + q.z**2))
    #roll = math.asin(2*(q.x*q.z - q.y*q.w))
    return -pitch + np.pi / 2
