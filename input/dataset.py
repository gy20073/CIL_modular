import random, cv2, time, threading, sys, Queue

import numpy as np
#from joblib import Parallel, delayed
from multiprocessing import Process, Pool
from multiprocessing import Queue as mQueue
import tensorflow as tf
from codification import *
from codification import encode, check_distance

from scipy.ndimage.filters import gaussian_filter
import copy
sys.path.append('utils')
import mapping_helper

from common_util import split_camera_middle_batch, camera_middle_zoom_batch

class Dataset(object):
    def __init__(self, splited_keys, images, datasets, config_input, augmenter, perception_interface):
        # sample inputs
        # splited_keys: _splited_keys_train[i_labels_per_division][i_steering_bins_perc][a list of keys]
        # images: [i_sensor][i_file_number] = (lastidx, lastidx + x.shape[0], x)
        # datasets: [i_target_name] = dim*batch matrix, where batch=#all_samples
        # config_input: configInputs
        # augmenter: config_input.augment

        # save the inputs
        self._splited_keys = splited_keys
        self._images = images
        self._targets = np.concatenate(tuple(datasets), axis=1)  # Cat the datasets, The shape is totalnum*totaldim
        self._config = config_input
        self._augmenter = augmenter

        self._batch_size = config_input.batch_size

        # prepare all the placeholders: 3 sources: _queue_image_input, _queue_targets, _queue_inputs
        self._queue_image_input = tf.placeholder(tf.float32, shape=[config_input.batch_size,
                                                                    config_input.feature_input_size[0],
                                                                    config_input.feature_input_size[1],
                                                                    config_input.feature_input_size[2]])

        self._queue_shapes = [self._queue_image_input.shape]

        # config.targets_names: ['wp1_angle', 'wp2_angle', 'Steer', 'Gas', 'Brake', 'Speed']
        self._queue_targets = []
        for i in range(len(self._config.targets_names)):
            self._queue_targets.append(tf.placeholder(tf.float32, shape=[config_input.batch_size,
                                                                         self._config.targets_sizes[i]]))
            self._queue_shapes.append(self._queue_targets[-1].shape)

        # self.inputs_names = ['Control', 'Speed']
        self._queue_inputs = []
        for i in range(len(self._config.inputs_names)):
            self._queue_inputs.append(tf.placeholder(tf.float32, shape=[config_input.batch_size,
                                                                        self._config.inputs_sizes[i]]))
            self._queue_shapes.append(self._queue_inputs[-1].shape)

        self._queue = tf.FIFOQueue(capacity=config_input.queue_capacity,
                                   dtypes=[tf.float32] + [tf.float32] * (len(self._config.targets_names) + len(self._config.inputs_names)),
                                   shapes=self._queue_shapes)
        self._enqueue_op = self._queue.enqueue([self._queue_image_input] + self._queue_targets + self._queue_inputs)
        self._dequeue_op = self._queue.dequeue()

        #self.parallel_workers = Parallel(n_jobs=8, backend="threading")
        self.input_queue = mQueue(5)
        self.output_queue = mQueue(5)

        self.perception_interface = perception_interface

        if "mapping" in self._config.inputs_names:
            version = "v1"
            if hasattr(self._config, "mapping_version"):
                version = self._config.mapping_version
            self.mapping_helper = mapping_helper.mapping_helper(output_height_pix=self._config.map_height,
                                                                version=version) # using the default values, 30 meters of width view, 50*75*1 output size


    def get_batch_tensor(self):
        return self._dequeue_op

    def sample_positions_to_train(self, number_of_samples, splited_keys):
        out_splited_keys = []
        for sp in splited_keys:
            if len(sp)>0:
                out_splited_keys.append(sp)

        return np.random.choice(range(len(out_splited_keys)),
                                size=number_of_samples,
                                replace=True), \
               out_splited_keys

    # Used by next_batch, for each of the control block,
    def datagen(self, batch_size, number_control_divisions):
        # typical input: batch_size, number_control_divisions=3, since 3 blocks
        # Goal: uniformly select from different control signals (group), different steering percentiles.
        generated_ids = np.zeros((batch_size, ), dtype='int32')

        count = 0
        to_be_decoded = [[] for _ in range(len(self._images))]
        for control_part in range(0, number_control_divisions):
            num_to_sample = int(batch_size // number_control_divisions)
            if control_part == (number_control_divisions - 1):
                num_to_sample = batch_size - (number_control_divisions - 1) * num_to_sample

            sampled_positions, non_empty_split_keys = self.sample_positions_to_train(num_to_sample,
                                                               self._splited_keys[control_part])

            for outer_n in sampled_positions:
                i = random.choice(non_empty_split_keys[outer_n])
                for isensor in range(len(self._images)):
                    # fetch the image from the h5 files
                    per_h5_len = self._images[isensor][0].shape[0]
                    ibatch = i // per_h5_len
                    iinbatch = i % per_h5_len
                    imencoded = self._images[isensor][ibatch][iinbatch]
                    to_be_decoded[isensor].append(imencoded)

                generated_ids[count] = i
                count += 1

        return to_be_decoded, generated_ids

    @staticmethod
    def get_boundary(oldseg):
        seg = copy.deepcopy(oldseg)
        seg[seg == 2] = 8
        ind7 = (seg == 7)
        hyp1 = np.logical_and(np.concatenate((seg[1:, :], np.zeros((1, seg.shape[1]))), axis=0) == 8, ind7)
        hyp2 = np.logical_and(np.concatenate((np.zeros((1, seg.shape[1])), seg[:-1, :]), axis=0) == 8, ind7)
        hyp3 = np.logical_and(np.concatenate((seg[:, 1:], np.zeros((seg.shape[0], 1))), axis=1) == 8, ind7)
        hyp4 = np.logical_and(np.concatenate((np.zeros((seg.shape[0], 1)), seg[:, :-1]), axis=1) == 8, ind7)
        lor = np.logical_or
        final = lor(lor(lor(hyp1, hyp2), hyp3), hyp4)

        e = gaussian_filter(final * 1.0, 10, mode='constant')
        e = (e * 1000).astype(np.uint8)
        e = (e > 0)

        e = np.logical_or(e, oldseg == 2)
        return e

    @staticmethod
    def augment_lane(camera, seg):
        camera = copy.deepcopy(camera)
        seg = copy.deepcopy(seg)
        seg = seg[:, :, 0]
        ind = (seg == 6)

        xs, ys = np.where(ind)
        ysp = np.minimum(ys + np.random.randint(-30, 30), camera.shape[1] - 1)
        camera[xs, ys, :] = camera[xs, ysp, :]

        bound = Dataset.get_boundary(seg)
        xs, ys = np.where(bound)
        ysp = ys + np.random.randint(-40, 40, size=ys.shape)
        ysp = np.minimum(camera.shape[1] - 1, np.maximum(0, ysp))
        camera[xs, ys, :] = camera[xs, ysp, :]

        return camera

    """Return the next `batch_size` examples from this data set."""

    # Used by enqueue
    def next_batch(self, sensors, generated_ids):
        # generate unbiased samples;
        # apply augmentation on sensors and segmentation labels
        # normalize images
        # fill in targets and inputs. with reasonable valid condition checking

        batch_size = self._batch_size

        if hasattr(self._config, "sensor_augments"):
            segmentations = sensors[len(sensors)//2 : ]
            sensors = sensors[:len(sensors) // 2]
            aug_ind = np.zeros(len(sensors[0]))
            for ib in range(len(sensors[0])):
                aug_ind[ib] = np.random.rand() < self._config.prob_augment_lane

        if self._augmenter[0] != None:
            aug_det = self._augmenter[0].to_deterministic()

        # Get the images -- Perform Augmentation!!!
        for i in range(len(sensors)):
            # decode each of the sensor in parallel
            func = lambda x: cv2.imdecode(x, 1)
            if hasattr(self._config, "hack_resize_image"):
                height, width = self._config.hack_resize_image
                func_previous = func
                func = lambda x: cv2.resize(func_previous(x), (width, height))

            if hasattr(self._config, "hack_faster_aug"):
                func_previous = func
                func = lambda x: func_previous(x)[::2, ::2, :]

            # func = delayed(func)
            # results = self.parallel_workers(func(x) for x in to_be_decoded[isensor])
            results = []
            for x in sensors[i]:
                results.append(func(x))
            sensors[i] = np.stack(results, 0)

            # from bgr to rgb
            sensors[i] = sensors[i][:, :, :, ::-1]

            if self._augmenter[i] != None:
                sensors[i] = aug_det.augment_images(sensors[i])

            # TODO add augmentation for lane markers and road boundary
            if hasattr(self._config, "prob_augment_lane") and self._augmenter[i]!=None:
                for ib in range(sensors[i].shape[0]):
                    if aug_ind[ib]:
                        if len(segmentations[i][ib]) > 0:
                            decoded = cv2.imdecode(segmentations[i][ib], 1)
                            sensors[i][ib, :, :, :] = self.augment_lane(sensors[i][ib, :,:,:], decoded)

                            if np.random.rand() < 0.005:
                                #cv2.imwrite("debug.png", sensors[i][ib, :,:,::-1])
                                pass

            if self._config.image_as_float[i]:
                sensors[i] = sensors[i].astype(np.float32)
            if self._config.sensors_normalize[i]:
                sensors[i] /= 255.0

        # TODO: perform image splitting here
        if hasattr(self._config, "camera_middle_split") and self._config.camera_middle_split:
            sensors = split_camera_middle_batch(sensors, self._config.sensor_names)
            if np.random.rand() < 0.05:
                pass
                '''
                print("debugging the camera split function")
                id = np.random.randint(0, sensors[0].shape[0])
                for i in range(len(sensors)):
                    cv2.imwrite("debug_%d.png" % i, sensors[i][id,:,:,::-1])
                '''

        if hasattr(self._config, "camera_middle_zoom"):
            sensors = camera_middle_zoom_batch(sensors, self._config.sensor_names, self._config.camera_middle_zoom)
            if np.random.rand() < 0.05:
                pass
                '''
                print("debugging the camera zoom function")
                id = np.random.randint(0, sensors[0].shape[0])
                for i in range(len(sensors)):
                    cv2.imwrite("debug_%d.png" % i, sensors[i][id,:,:,::-1])
                '''


        # self._targets is the targets variables concatenated
        # Get the targets

        # merge the follow and the straights in targets
        k = self._config.variable_names.index('Control')
        self._targets[(self._targets[:, k]).astype(np.int) == 5, k] = 2.0
        # end of the merging

        target_selected = self._targets[generated_ids, :]
        target_selected = target_selected.T

        # prepare the output targets, and inputs
        targets = []
        for i in range(len(self._config.targets_names)):
            targets.append(np.zeros((batch_size, self._config.targets_sizes[i])))
        inputs = []
        for i in range(len(self._config.inputs_names)):
            inputs.append(np.zeros((batch_size, self._config.inputs_sizes[i])))

        for ibatch in range(0, batch_size):
            for itarget in range(len(self._config.targets_names)):
                # Yang: This is assuming that all target names has size 1
                k = self._config.variable_names.index(self._config.targets_names[itarget])
                targets[itarget][ibatch] = target_selected[k, ibatch]
                this_name = self._config.targets_names[itarget]

                if this_name == "Speed":
                    # Yang: speed_factor is normalizing the speed
                    targets[itarget][ibatch] /= self._config.speed_factor / 3.6
                elif this_name == "Gas":
                    # Yang: require Gas >=0
                    targets[itarget][ibatch] = max(0, targets[itarget][ibatch])
                elif this_name == "Brake":
                    # Yang: require 0<=Brake<=1
                    targets[itarget][ibatch] = min(1.0, max(0, targets[itarget][ibatch]))

            for iinput in range(len(self._config.inputs_names)):
                this_name = self._config.inputs_names[iinput]

                if this_name == "mapping":
                    # make the map
                    pos_x = self._config.variable_names.index("Pos_X")
                    pos_y = self._config.variable_names.index("Pos_Y")
                    ori_x = self._config.variable_names.index("Ori_X")
                    ori_y = self._config.variable_names.index("Ori_Y")
                    ori_z = self._config.variable_names.index("Ori_Z")
                    town_id = self._config.variable_names.index("town_id")

                    pos = [target_selected[pos_x, ibatch], target_selected[pos_y, ibatch]]
                    ori = [target_selected[ori_x, ibatch], target_selected[ori_y, ibatch], target_selected[ori_z, ibatch]]
                    town_id = int(target_selected[town_id, ibatch])
                    town_id =str(town_id).zfill(2)
                    if self._augmenter[0]!=None:
                        # we are in the training mode, thus we need to add some noise to the position
                        std = self._config.map_pos_noise_std
                        fun_trunc_normal = lambda std: max(min(np.random.normal(scale=std), 2*std), -2*std)
                        pos = [pos[0] + fun_trunc_normal(std), pos[1] + fun_trunc_normal(std)]

                        if town_id == "01" or town_id == "02":
                            # noise to the yaw
                            yaw = np.arctan2(-ori[1], ori[0]) + np.random.normal(scale=np.deg2rad(self._config.map_yaw_noise_std))
                            ori[0] = np.cos(yaw)
                            ori[1] = - np.sin(yaw)
                        elif town_id == "10" or town_id == "11" or town_id == "13":
                            ori[2] += np.rad2deg(np.random.normal(scale=np.deg2rad(self._config.map_yaw_noise_std)))
                        else:
                            raise ValueError()

                    map = self.mapping_helper.get_map(town_id, pos, ori)
                    # add a flattened operator, to make it compatible with the original format, remember to reshape it back
                    inputs[iinput][ibatch] = map.flatten()

                    if np.random.rand() < 0.005:
                        # for debugging
                        pass
                        '''
                        im = self.mapping_helper.map_to_debug_image(map)[:,:,::-1]
                        center = sensors[1][ibatch, :,:,::-1]
                        cv2.imwrite("debug_map.png", im)
                        cv2.imwrite("debug_center_cam.png", center)
                        '''
                elif this_name == "dis_to_road_border":
                    pos_x = self._config.variable_names.index("Pos_X")
                    pos_y = self._config.variable_names.index("Pos_Y")
                    ori_x = self._config.variable_names.index("Ori_X")
                    ori_y = self._config.variable_names.index("Ori_Y")
                    ori_z = self._config.variable_names.index("Ori_Z")
                    town_id = self._config.variable_names.index("town_id")

                    pos = [target_selected[pos_x, ibatch], target_selected[pos_y, ibatch]]
                    ori = [target_selected[ori_x, ibatch], target_selected[ori_y, ibatch],
                           target_selected[ori_z, ibatch]]
                    town_id = int(target_selected[town_id, ibatch])
                    town_id = str(town_id).zfill(2)
                    map = self.mapping_helper.get_map(town_id, pos, ori)

                    # add a flattened operator, to make it compatible with the original format, remember to reshape it back
                    inputs[iinput][ibatch] = self.mapping_helper.compute_dis_to_border(map)
                    #again
                elif this_name == "is_onroad":
                    pos_x = self._config.variable_names.index("Pos_X")
                    pos_y = self._config.variable_names.index("Pos_Y")
                    ori_x = self._config.variable_names.index("Ori_X")
                    ori_y = self._config.variable_names.index("Ori_Y")
                    ori_z = self._config.variable_names.index("Ori_Z")
                    town_id = self._config.variable_names.index("town_id")

                    pos = [target_selected[pos_x, ibatch], target_selected[pos_y, ibatch]]
                    ori = [target_selected[ori_x, ibatch], target_selected[ori_y, ibatch],
                           target_selected[ori_z, ibatch]]
                    town_id = int(target_selected[town_id, ibatch])
                    town_id = str(town_id).zfill(2)
                    map = self.mapping_helper.get_map(town_id, pos, ori)

                    # add a flattened operator, to make it compatible with the original format, remember to reshape it back
                    inputs[iinput][ibatch] = self.mapping_helper.is_on_road(map)
                    #again
                elif this_name == "is_onshoulder":
                    pos_x = self._config.variable_names.index("Pos_X")
                    pos_y = self._config.variable_names.index("Pos_Y")
                    ori_x = self._config.variable_names.index("Ori_X")
                    ori_y = self._config.variable_names.index("Ori_Y")
                    ori_z = self._config.variable_names.index("Ori_Z")
                    town_id = self._config.variable_names.index("town_id")

                    pos = [target_selected[pos_x, ibatch], target_selected[pos_y, ibatch]]
                    ori = [target_selected[ori_x, ibatch], target_selected[ori_y, ibatch],
                           target_selected[ori_z, ibatch]]
                    town_id = int(target_selected[town_id, ibatch])
                    town_id = str(town_id).zfill(2)
                    map = self.mapping_helper.get_map(town_id, pos, ori)

                    color = map[map.shape[0]*3//4, map.shape[1]//2, :]
                    # TODO: it has to use map v2, not v3
                    assert self.mapping_helper.version == "v2"
                    if color[0] == 1 and color[1] == 0 and color[2] == 0:
                        on_shoulder = 1
                    else:
                        on_shoulder = 0
                    inputs[iinput][ibatch] = on_shoulder
                else:
                    k = self._config.variable_names.index(self._config.inputs_names[iinput])
                    if this_name == "Control":
                        inputs[iinput][ibatch] = encode(target_selected[k, ibatch])
                    elif this_name == "Speed":
                        inputs[iinput][ibatch] = target_selected[k, ibatch] / self._config.speed_factor * 3.6
                    elif this_name == "Distance":
                        inputs[iinput][ibatch] = check_distance(target_selected[k, ibatch])
                    else:
                        raise ValueError()

        # change the output sensors variable
        sensors = np.concatenate(sensors, axis=0)
        if self._augmenter[0] != None and hasattr(self._config, "sensor_dropout") and self._config.sensor_dropout > 0:
            # do the sensor dropout
            # sensors[i] B H W C, after concat is 3B H W C
            #print("augmenting the sensors dropout")
            num_images = sensors.shape[0]
            for i in range(num_images):
                if np.random.rand() < self._config.sensor_dropout:
                    sensors[i, :, :, :] = np.mean(sensors[i, :, :, :])

            if "mapping" in self._config.inputs_names:
                #print("augmenting the mapping dropout")
                id = self._config.inputs_names.index("mapping")
                for i in range(inputs[id].shape[0]):
                    if np.random.rand() < self._config.mapping_dropout:
                        inputs[id][i, :] = np.mean(inputs[id][i])


        return sensors, targets, inputs

    @staticmethod
    def random_region_old(H, W, size, diversity_prob):
        image = np.zeros((H, W), dtype=np.bool)
        openlist = []

        def check_neighbours(image, point):
            x, y = point
            res = []

            for direction in ((-1, 0), (1, 0), (0, 1), (0, -1)):
                newx = x + direction[0]
                newy = y + direction[1]
                if newx >= 0 and newx < image.shape[0] and newy >= 0 and newy < image.shape[1]:
                    if image[newx, newy] == 0:
                        res.append((newx, newy))
            return res

        i = 0
        while i < size:
            if openlist == []:
                x = np.random.randint(0, H)
                y = np.random.randint(0, W)
                openlist.append((x, y))
                image[x, y] = 1
                i += 1
            else:
                if np.random.rand() < diversity_prob:
                    x = np.random.randint(0, H)
                    y = np.random.randint(0, W)
                    if image[x, y] == 0:
                        openlist.append((x, y))
                        image[x, y] = 1
                        i += 1
                        continue

                # random a point from the open list
                j = np.random.randint(0, len(openlist))
                point = openlist[j]
                res = check_neighbours(image, point)
                if len(res) == 0:
                    openlist = openlist[0:j] + openlist[j + 1:]
                else:
                    nex = random.choice(res)
                    openlist.append(nex)
                    image[nex[0], nex[1]] = 1
                    i += 1
        return image

    @staticmethod
    def random_region(H, W, un, un2):
        image = np.zeros((H, W), dtype=np.bool)
        # from 1/3 expectation 2 to minimum 3 in shape, each time
        sizeW = W // 3
        nregions = 1.5
        while sizeW >= 3:
            sizeH = int(sizeW * 1.0 / W * H)
            for i in range(np.random.poisson(nregions)):
                this_w = sizeW + int(np.random.normal(0, sizeW // 4))
                this_h = sizeH + int(np.random.normal(0, sizeH // 4))
                left_up_h = np.random.randint(0, H - this_h)
                left_up_w = np.random.randint(0, W - this_w)
                image[left_up_h:(left_up_h + this_h), left_up_w:(left_up_w + this_w)] = True

            sizeW = sizeW // 2
            nregions *= 2
        return image

    def process_run(self, sess, data_loaded):
        t00 = time.time()
        reshaped = data_loaded[0]
        nB, nH, nW, nC = reshaped.shape
        num_sensors = len(self._config.sensor_names)
        if hasattr(self._config, "camera_middle_split") and self._config.camera_middle_split:
            num_sensors += 1

        t0 = time.time()
        reshaped = np.reshape(reshaped, (num_sensors, nB//num_sensors, nH, nW, nC))
        if (not hasattr(self._config, "camera_combine")) or self._config.camera_combine == "width_stack":
            reshaped = np.transpose(reshaped, (1, 2, 0, 3, 4))
            # now has shape nB//num_sensors, nH, num_sensors, nW, nC
            reshaped = np.reshape(reshaped, (nB//num_sensors, nH, num_sensors*nW, nC))
            print("width stack")
            #print("width stack cost ", time.time() - t0)
        elif self._config.camera_combine == "channel_stack":
            reshaped = np.transpose(reshaped, (1, 2, 3, 4, 0))
            # now has shape nB//num_sensors, nH, nW, nC, num_sensors
            reshaped = np.reshape(reshaped, (nB // num_sensors, nH, nW, nC * num_sensors))
            #print("channel stack")
            #print("shape of channel stack", reshaped.shape)
            # print("channel stack total cost ", time.time() - t0)


        if hasattr(self._config, "add_gaussian_noise") and self._augmenter[0]!=None:
            std = self._config.add_gaussian_noise
            print("!!!!!!!!!!!!!!!!!!adding gaussian noise", std)
            reshaped += np.random.normal(0, std, reshaped.shape)

        t0 = time.time()
        if hasattr(self._config, "add_random_region_noise") and self._augmenter[0] != None:
            std = self._config.add_random_region_noise
            #print("!!!!!!!!!!!!!!!!!add random region noise", std)
            mask = Dataset.batch_random_region(reshaped.shape[0], reshaped.shape[1], reshaped.shape[2])
            mask = np.reshape(mask, (reshaped.shape[0], reshaped.shape[1], reshaped.shape[2], 1))
            reshaped += np.random.normal(0, std, (reshaped.shape[0], 1, 1, reshaped.shape[3])) * mask
            #print("random region noise total cost ", time.time() - t0)

        queue_feed_dict = {self._queue_image_input: reshaped}  # images we already put by default

        for i in range(len(self._config.targets_names)):
            queue_feed_dict.update({self._queue_targets[i]: data_loaded[1][i]})

        for i in range(len(self._config.inputs_names)):
            queue_feed_dict.update({self._queue_inputs[i]: data_loaded[2][i]})

        t0 = time.time()
        sess.run(self._enqueue_op, feed_dict=queue_feed_dict)
        #print("feed dict cost ", time.time() - t0)

        #print("total in process run cost ", time.time() - t00)

    # Used by enqueue
    @staticmethod
    def batch_random_region(B, H, W):
        ans = []
        for i in range(B):
            size = np.random.randint(H*W//8, H*W//3)
            ans.append(Dataset.random_region(H, W, size, 0.05))
        return np.stack(ans, axis=0)

    def __getstate__(self):
        """Return state values to be pickled."""
        print("pickling")
        return (self._splited_keys, self._targets, self._config, self._augmenter, self._batch_size)

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        print("unpickling")
        self._splited_keys, self._targets, self._config, self._augmenter, self._batch_size = state

    def _thread_disk_reader(self):
        while True:
            #start = time.time()
            sensors, generated_ids = self.datagen(self._batch_size, len(self._splited_keys))
            self.input_queue.put((sensors, generated_ids))
            #print('putting ele into input queue, cost', time.time()-start)

    @staticmethod
    def _thread_decode_augment(dataset, input_queue, output_queue):
        while True:
            sensors, generated_ids = input_queue.get()
            out = dataset.next_batch(sensors, generated_ids)
            output_queue.put(out)

    def start_multiple_decoders_augmenters(self):
        n_jobs = 6
        for i in range(n_jobs):
            p = Process(target=self._thread_decode_augment, args=(self, self.input_queue, self.output_queue))
            #p = threading.Thread(target=self._thread_decode_augment, args=(self, self.input_queue, self.output_queue))
            p.start()

    def _thread_perception_splitting(self, input_queue):
        while True:
            one_batch = input_queue.get()
            print("input queue qsize", input_queue.qsize())
            self.output_remaining_queue.put(one_batch[1:])
            self.output_image_queue.put(one_batch[0])

    def _thread_perception_concat(self, perception_output):
        while True:
            remain = self.output_remaining_queue.get()
            image_feature = perception_output.get()
            self.final_output_queue.put([image_feature, remain[0], remain[1]])

    def _thread_feed_dict(self, sess, output_queue):
        while True:
            #start = time.time()
            one_batch = output_queue.get()
            print("output qsize is", output_queue.qsize())
            self.process_run(sess, one_batch)
            #print("fetched one output, cost ", time.time()-start)

    def start_all_threads(self, sess):
        t = threading.Thread(target=self._thread_disk_reader)
        t.isDaemon()
        t.start()

        self.start_multiple_decoders_augmenters()

        if self._config.use_perception_stack:
            self.output_image_queue = Queue.Queue(5)
            self.output_remaining_queue = Queue.Queue(5)
            t = threading.Thread(target=self._thread_perception_splitting, args=(self.output_queue,))
            t.start()

            perception_output = self.perception_interface.compute_async_thread_channel(self.output_image_queue)

            self.final_output_queue = Queue.Queue(5)
            t = threading.Thread(target=self._thread_perception_concat, args=(perception_output,))
            t.start()
            output_queue = self.final_output_queue
        else:
            output_queue = self.output_queue

        t = threading.Thread(target=self._thread_feed_dict, args=(sess, output_queue))
        t.isDaemon()
        t.start()
