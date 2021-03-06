import glob, os
from imgaug import augmenters as iaa

class configMain:
    def __init__(self):
        self.steering_bins_perc = [0.05, 0.05, 0.1, 0.3, 0.3, 0.1, 0.05, 0.05]
        self.number_steering_bins = len(self.steering_bins_perc)
        self.batch_size = self.number_steering_bins * 15
        self.batch_size_val = self.number_steering_bins * 15
        self.number_images_val = self.batch_size_val  # Number of images used in a validation Section - Default: 20*

        self.image_size = (88, 200, 3)
        # the speed unit has changed, when reading from the h5 file, we change the speed_factor to adjust for this
        self.variable_names = ['Steer', 'Gas', 'Brake', 'Hand_B', 'Reverse',
                               'Steer_N', 'Gas_N', 'Brake_N',
                               'Pos_X', 'Pos_Y', 'Speed',
                               'C_Gen', 'C_Ped', 'C_Car', 'Road_I', 'Side_I', 'Acc_x', 'Acc_y', 'Acc_z',
                               'Plat_Ts', 'Game_Ts', 'Ori_X', 'Ori_Y', 'Ori_Z', 'Control', 'Camera', 'Angle',
                               'wp1_x', 'wp1_y', 'wp2_x', 'wp2_y', 'wp1_angle', 'wp1_mag', 'wp2_angle', 'wp2_mag']

        self.sensor_names = ['CameraMiddle']
        self.sensors_normalize = [True]

        self.targets_names = ['wp1_angle', 'wp2_angle', 'Steer', 'Gas', 'Brake', 'Speed']
        self.targets_sizes = [1, 1, 1, 1, 1, 1]

        self.inputs_names = ['Control', 'Speed']
        self.inputs_sizes = [4, 1]

        # if there is branching, this is used to build the network. Names should be same as targets
        # currently the ["Steer"]x4 should not be changed
        self.branch_config = [['wp1_angle', 'wp2_angle', "Steer", "Gas", "Brake"],
                              ['wp1_angle', 'wp2_angle', "Steer", "Gas", "Brake"],
                              ['wp1_angle', 'wp2_angle', "Steer", "Gas", "Brake"],
                              ['wp1_angle', 'wp2_angle', "Steer", "Gas", "Brake"], ["Speed"]]

        # a list of keep_prob corresponding to the list of layers:
        # 8 conv layers, 2 img FC layer, 2 speed FC layers, 1 joint FC layer, 5 branches X 2 FC layers each
        self.dropout = [0.8] * 8 + [0.5] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] * len(self.branch_config)

        self.models_path = os.path.join('models', os.path.basename(__file__).split('.')[0])
        self.train_path_write = os.path.join(self.models_path, 'train')
        self.val_path_write = os.path.join(self.models_path, 'val')

        self.number_iterations = 501000  # 500k
        self.perform_simulation_test = False
        self.output_is_on = True

        # enable the segmentation model
        self.segmentation_model = None
        #self.segmentation_model_name = "ErfNet_Small"

        self.use_perception_stack = False
        self.feature_input_size = self.image_size

        self.optimizer = "adam"

class configInput(configMain):
    def __init__(self):
        configMain.__init__(self)

        st = lambda aug: iaa.Sometimes(0.2, aug)
        oc = lambda aug: iaa.Sometimes(0.1, aug)
        rl = lambda aug: iaa.Sometimes(0.05, aug)

        self.augment = [iaa.Sequential([
            rl(iaa.GaussianBlur((0, 1.3))),  # blur images with a sigma between 0 and 1.5
            rl(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05), per_channel=0.5)),  # add gaussian noise to images
            rl(iaa.Dropout((0.0, 0.10), per_channel=0.5)),  # randomly remove up to X% of the pixels
            oc(iaa.Add((-20, 20), per_channel=0.5)),  # change brightness of images (by -X to Y of original value)
            st(iaa.Multiply((0.25, 2.5), per_channel=0.2)),  # change brightness of images (X-Y% of original value)
            rl(iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)),  # improve or worsen the contrast
            rl(iaa.Grayscale((0.0, 1))),  # put grayscale
            ],
            random_order=True  # do all of the above in random order
        )]

        self.val_db_path = glob.glob("/data/yang/code/aws/scratch/carla_collect/matthias_nosteeraug/val/*.h5")
        self.train_db_path = glob.glob("/data/yang/code/aws/scratch/carla_collect/matthias_nosteeraug/train/*.h5")

        self.speed_factor = 40.0  # In KM/H, the new measurement unit is in m/s, thus we had to change the factor

        # The division is made by three diferent data kinds
        # in every mini-batch there will be equal number of samples with labels from each group
        # e.g. for [[0,1],[2]] there will be 50% samples with labels 0 and 1, and 50% samples with label 2
        self.labels_per_division = [[0, 2, 5], [3], [4]]
        self.dataset_names = ['targets']
        self.queue_capacity = 5 # now measured in how many batches

        # TODO: move from this hacky way of resizing to something more systematic
        self.hack_resize_image = (88, 200)
        self.image_as_float = [True]

    # TODO NOT IMPLEMENTED Felipe: True/False switches to turn data balancing on or off


class configTrain(configMain):
    def __init__(self):
        configMain.__init__(self)

        self.loss_function = 'mse_branched'  # Chose between: mse_branched, mse_branched_ladd
        self.control_mode = 'single_branch_wp'
        self.learning_rate = 0.0002
        # use the default segmentation network
        #self.seg_network_erfnet_one_hot = True  # comment this line out to use the standard network
        #self.train_segmentation = True
        #self.number_of_labels = 2

        self.training_schedule = [[50000, 0.5], [100000, 0.5 * 0.5], [150000, 0.5 * 0.5 * 0.5],
                                  [200000, 0.5 * 0.5 * 0.5 * 0.5],
                                  [250000, 0.5 * 0.5 * 0.5 * 0.5 * 0.5]]  # Number of iterations, multiplying factor

        self.branch_loss_weight = [0.95, 0.95, 0.95, 0.95, 0.05]
        self.variable_weight = {'wp1_angle': 0.3, 'wp2_angle': 0.3, 'Steer': 0.1, 'Gas': 0.2, 'Brake': 0.1, 'Speed': 1.0}
        self.network_name = 'chauffeurNet_deeper'
        self.is_training = True


class configOutput(configMain):
    def __init__(self):
        configMain.__init__(self)
        self.print_interval = 10 # how often in output manager to print
        self.summary_writing_period = 100 # for output manager
        self.validation_period = 1000  # For output manager, I consider validation as an output thing since it does not directly affects the training in general



# The class below is deprecated, but save it for now, because we have to migrate the testing process to the new benchmark.
class configTest(configMain):
    def __init__(self):
        configMain.__init__(self)
        self.driver_config = "9cam_agent_carla_test_rc"  # only used, when calling the drive() function
