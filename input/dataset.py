import random, cv2

import numpy as np
import tensorflow as tf
from codification import *
from codification import encode, check_distance

class Dataset(object):
    def __init__(self, splited_keys, images, datasets, config_input, augmenter):
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
                                                                    config_input.image_size[0],
                                                                    config_input.image_size[1],
                                                                    config_input.image_size[2]])

        self._queue_shapes = [[config_input.batch_size, config_input.image_size[0], config_input.image_size[1], config_input.image_size[2]]]

        # config.targets_names: ['wp1_angle', 'wp2_angle', 'Steer', 'Gas', 'Brake', 'Speed']
        self._queue_targets = []
        for i in range(len(self._config.targets_names)):
            self._queue_targets.append(tf.placeholder(tf.float32, shape=[config_input.batch_size,
                                                                         self._config.targets_sizes[i]]))
            self._queue_shapes.append([config_input.batch_size, self._config.targets_sizes[i]])

        # self.inputs_names = ['Control', 'Speed']
        self._queue_inputs = []
        for i in range(len(self._config.inputs_names)):
            self._queue_inputs.append(tf.placeholder(tf.float32, shape=[config_input.batch_size,
                                                                        self._config.inputs_sizes[i]]))
            self._queue_shapes.append([config_input.batch_size, self._config.inputs_sizes[i]])

        self._queue = tf.FIFOQueue(capacity=config_input.queue_capacity,
                                   dtypes=[tf.float32] * (len(self._config.targets_names) + len(self._config.inputs_names) + 1),
                                   shapes=self._queue_shapes)
        self._enqueue_op = self._queue.enqueue([self._queue_image_input] + self._queue_targets + self._queue_inputs)
        self._dequeue_op = self._queue.dequeue()

    def get_batch_tensor(self):
        return self._dequeue_op

    def sample_positions_to_train(self, number_of_samples):
        return np.random.choice(range(self._config.number_steering_bins),
                                size=number_of_samples,
                                replace=True)

    # Used by next_batch, for each of the control block,
    def datagen(self, batch_size, number_control_divisions):
        # typical input: batch_size, number_control_divisions=3, since 3 blocks
        # Goal: uniformly select from different control signals (group), different steering percentiles.
        sensors_batch = []
        for i in range(len(self._images)):
            # typical config.sensors_size = [(88, 200, 3)]
            sensors_batch.append(np.zeros((batch_size,
                                           self._config.sensors_size[i][0],
                                           self._config.sensors_size[i][1],
                                           self._config.sensors_size[i][2]), dtype='uint8'))
        generated_ids = np.zeros((batch_size, ), dtype='int32')

        count = 0
        for control_part in range(0, number_control_divisions):
            num_to_sample = int(batch_size // number_control_divisions)
            if control_part == (number_control_divisions - 1):
                num_to_sample = batch_size - (number_control_divisions - 1) * num_to_sample

            sampled_positions = self.sample_positions_to_train(num_to_sample)

            for outer_n in sampled_positions:
                i = random.choice(self._splited_keys[control_part][outer_n])
                for isensor in range(len(self._images)):
                    # fetch the image from the h5 files
                    per_h5_len = self._images[isensor][0].shape[0]
                    ibatch = i // per_h5_len
                    iinbatch = i % per_h5_len
                    imencoded = self._images[isensor][ibatch][iinbatch]
                    # decode the image
                    decoded = cv2.imdecode(imencoded, 1)
                    if hasattr(self._config, "hack_resize_image"):
                        decoded = cv2.resize(decoded, self._config.hack_resize_image)
                    sensors_batch[isensor][count, :, :, :] = decoded

                generated_ids[count] = i
                count += 1
        return sensors_batch, generated_ids

    """Return the next `batch_size` examples from this data set."""

    # Used by enqueue
    def next_batch(self):
        # generate unbiased samples;
        # apply augmentation on sensors and segmentation labels
        # normalize images
        # fill in targets and inputs. with reasonable valid condition checking

        batch_size = self._batch_size
        sensors, generated_ids = self.datagen(batch_size, len(self._splited_keys))

        # Get the images -- Perform Augmentation!!!
        for i in range(len(sensors)):
            if self._augmenter[i] != None:
                sensors[i] = self._augmenter[i].augment_images(sensors[i])
            sensors[i] = sensors[i].astype(np.float32)

        # self._targets is the targets variables concatenated
        # Get the targets
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
            for isensor in range(len(self._images)):  # number sensors
                if self._config.sensors_normalize[isensor]:
                    sensors[isensor][ibatch, :, :, :] = np.multiply(sensors[isensor][ibatch, :, :, :], 1.0 / 255.0)

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
                k = self._config.variable_names.index(self._config.inputs_names[iinput])
                this_name = self._config.inputs_names[iinput]

                if this_name == "Control":
                    inputs[iinput][ibatch] = encode(target_selected[k, ibatch])
                elif this_name == "Speed":
                    inputs[iinput][ibatch] = target_selected[k, ibatch] / self._config.speed_factor * 3.6
                elif this_name == "Distance":
                    inputs[iinput][ibatch] = check_distance(target_selected[k, ibatch])
                else:
                    raise ValueError()

        assert(self._config.sensor_names == ['CameraMiddle'])
        return sensors[self._config.sensor_names.index('CameraMiddle')], targets, inputs

    # Used by enqueue
    def process_run(self, sess, data_loaded):
        queue_feed_dict = {self._queue_image_input: data_loaded[0]}  # images we already put by default

        for i in range(len(self._config.targets_names)):
            queue_feed_dict.update({self._queue_targets[i]: data_loaded[1][i]})

        for i in range(len(self._config.inputs_names)):
            queue_feed_dict.update({self._queue_inputs[i]: data_loaded[2][i]})

        sess.run(self._enqueue_op, feed_dict=queue_feed_dict)

    # the main entrance from other process
    def enqueue(self, sess):
        while True:
            # data_loaded[0] is the images, [1] is the target_names [2] is the input names
            data_loaded = self.next_batch()
            # process run enqueue the prepared dict
            self.process_run(sess, data_loaded)
