import sys, h5py, threading
import tensorflow as tf

sys.path.append('spliter')
from spliter import Spliter
from dataset import *

# TODO: Divide also by acceleration and Steering

class DatasetManager(object):
    def __init__(self, config):
        self._images_train, self._datasets_train = self.read_all_files(config.train_db_path,
                                                                       config.sensor_names,
                                                                       config.dataset_names)
        self._images_val, self._datasets_val = self.read_all_files(config.val_db_path,
                                                                   config.sensor_names,
                                                                   config.dataset_names)

        spliter = Spliter(1, 1, config.steering_bins_perc)

        #self.labels_per_division = [[0, 2, 5], [3], [4]]
        divided_keys_train = spliter.divide_keys_by_labels(
                self._datasets_train[0][config.variable_names.index("Control")][:],
                config.labels_per_division)
        # The structure is: self._splited_keys_train[i_labels_per_division][i__steering_bins_perc][a list of keys]
        # In theory should be sharded instance ids by those two criterions
        self._splited_keys_train = spliter.split_by_output(
                self._datasets_train[0][config.variable_names.index("Steer")][:],
                divided_keys_train)

        divided_keys_val = spliter.divide_keys_by_labels(
                self._datasets_val[0][config.variable_names.index("Control")][:],
                config.labels_per_division)  # THE NOISE IS NOW NONE, TEST THIS
        self._splited_keys_val = spliter.split_by_output(
                self._datasets_val[0][config.variable_names.index("Steer")][:],
                divided_keys_val)

        print("max id train", max(max(max(self._splited_keys_train))))
        print("min id train", min(min(min(self._splited_keys_train))))
        print("max id val", max(max(max(self._splited_keys_val))))

        self.train = Dataset(self._splited_keys_train,
                             self._images_train,
                             self._datasets_train, config, config.augment)
        self.validation = Dataset(self._splited_keys_val,
                                  self._images_val,
                                  self._datasets_val, config, [None] * len(config.sensor_names))

    def start_training_queueing(self, sess):
        # TODO: those extra threading for enqueue operation might be unnecessary
        enqueue_thread = threading.Thread(target=self.train.enqueue, args=[sess])
        enqueue_thread.isDaemon()
        enqueue_thread.start()

        coord = tf.train.Coordinator()
        self._threads_train = tf.train.start_queue_runners(coord=coord, sess=sess)

    def start_validation_queueing(self, sess):
        enqueue_thread = threading.Thread(target=self.validation.enqueue, args=[sess])
        enqueue_thread.isDaemon()
        enqueue_thread.start()

        coord = tf.train.Coordinator()
        self._threads_val = tf.train.start_queue_runners(coord=coord, sess=sess)

    def read_all_files(self, file_names, sensor_names, target_names):
        sensor_cat = [list([]) for _ in range(len(sensor_names))]
        targets_cat = [list([]) for _ in range(len(target_names))]

        lastidx = 0
        for cword in file_names:
            try:
                dset = h5py.File(cword, "r")
                for i in range(len(sensor_names)):
                    x = dset[sensor_names[i]]
                    old_shape = x.shape[0]
                    sensor_cat[i].append((lastidx, lastidx + x.shape[0], x))

                for i in range(len(target_names)):
                    dset_to_append = dset[target_names[i]]
                    targets_cat[i].append(dset_to_append[:])

                lastidx += old_shape
                dset.flush()

            except IOError:
                import traceback
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exc()
                traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=2, file=sys.stdout)
                print("failed to open", cword)

        for i in range(len(target_names)):
            targets_cat[i] = np.concatenate(targets_cat[i], axis=0)
            targets_cat[i] = targets_cat[i].transpose((1, 0))

        # sensor_cat is a list for each of the variables, each of them is a list of (start_index, end_index, x) tuples
        # targets_cat is a list for each of the variables, variable across batch are concatenated together and transposed to
        #       have size dim*batch
        return sensor_cat, targets_cat
