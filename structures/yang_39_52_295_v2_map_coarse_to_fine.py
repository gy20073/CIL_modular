import numpy as np
from network import Network


def mapping_tower(mapping, network_manager, tf, scope_name="mapping_tower", channel_multiplier=1):
    cm = channel_multiplier
    # TODO: change the mapping tower dropout rates in the config file
    with tf.variable_scope(scope_name):
        print("within mapping tower")
        x = mapping
        '''dimension reduction'''
        # size 39*52*295
        xc = network_manager.conv_block(x, 1, 1, 32*cm, padding_in='VALID')
        # size 39*52*64
        print(xc)
        """conv1"""  # kernel sz, stride, num feature maps
        xc = network_manager.conv_block(xc, 3, 1, 32*cm, padding_in='VALID')
        # size 37*50*64
        print(xc)

        """conv2"""
        xc = network_manager.conv_block(xc, 3, 2, 64*cm, padding_in='VALID')
        # size 18*24*128
        print(xc)
        xc = network_manager.conv_block(xc, 3, 1, 64*cm, padding_in='VALID')
        # size 16*22*128
        print(xc)

        """conv3"""
        xc = network_manager.conv_block(xc, 3, 2, 128*cm, padding_in='VALID')
        # size 7*10*256
        print(xc)
        xc = network_manager.conv_block(xc, 3, 1, 128*cm, padding_in='VALID')
        # size 5*8*256
        print(xc)

        """conv4"""
        xc = network_manager.conv_block(xc, 3, 2, 256*cm, padding_in='VALID')
        # size 2*3*512
        print(xc)
        xc = tf.reduce_mean(xc, axis=[1, 2], keep_dims=True)
        print(xc)
        """mp3 (default values)"""

        """ reshape """
        x = tf.reshape(xc, [-1, int(np.prod(xc.get_shape()[1:]))], name='reshape')
        print(x)

        """ fc1 """
        x = network_manager.fc_block(x, 256*cm)
        print(x)
        """ fc2 """
        x = network_manager.fc_block(x, 256*cm)
        print("before exiting the mapping tower")
        return x

def make_branches(branch_config, input_feature, input_feature_with_speed, tf, network_manager, scope_name, branched_features=None):
    branches = []
    """Start BRANCHING"""
    with tf.variable_scope(scope_name):
        for i in range(0, len(branch_config)):
            with tf.name_scope("Branch_" + str(i)):
                if branch_config[i][0] == "Speed":
                    main_feature = input_feature
                else:
                    main_feature = input_feature_with_speed

                if branched_features is not None and len(branched_features) > i:
                    extra = branched_features[i]
                    extra = network_manager.fc_block(extra, 128)
                    extra = network_manager.fc_block(extra, 128)
                    main_feature = tf.concat([main_feature, extra], 1)

                branch_output = network_manager.fc_block(main_feature, 256)
                branch_output = network_manager.fc_block(branch_output, 256)

                branches.append(network_manager.fc(branch_output, len(branch_config[i])))
            print(branch_output)
    return branches

def add_map_and_image_delta_with_threshold(branches_map, branches_image_delta, branches_map_sigma, config_image, config_map, tf):
    # config_image should contain all required output, config_map could only contains a subset
    branches = []
    for i in range(len(config_image)):
        this_list = []
        for j in range(len(config_image[i])):
            if len(config_map)>i and config_image[i][j] in config_map[i]:
                index = config_map[i].index(config_image[i][j])
                correction = tf.maximum(branches_image_delta[i][:, j], -tf.abs(branches_map_sigma[i][:, index]))
                correction = tf.minimum(correction            ,  tf.abs(branches_map_sigma[i][:, index]))

                out = branches_map[i][:, j] + correction
            else:
                out = branches_image_delta[i][:, j]
            this_list.append(out)
        branches.append(tf.stack(this_list, 1))
    return branches


def create_structure(tf, input_image, input_data, input_size, dropout, config):
    # has a few losses: the ma
    x = input_image
    mapping = input_data[config.inputs_names.index("mapping")]
    # now have shape batchsize*(50*75)
    mapping = tf.reshape(mapping, [-1, config.map_height, config.map_height * 3 // 2, 1])
    mapping = tf.image.resize_bilinear(mapping, [39, 52], name="resize_mapping")

    network_manager = Network(config, dropout, tf.shape(x))

    mapping_feature = mapping_tower(mapping, network_manager, tf)
    # branch_config_map should be like [['Steer']] * 4, since in practice we only care about the steer
    branches_map = make_branches(config.branch_config_map, mapping_feature, mapping_feature, tf, network_manager, "map_prediction")
    branches_map_sigma = make_branches(config.branch_config_map, mapping_feature, mapping_feature, tf, network_manager, "map_prediction_sigma")

    image_feature = mapping_tower(input_image, network_manager, tf, "image_tower", channel_multiplier=2)

    """ Speed (measurements)"""
    with tf.name_scope("Speed"):
        speed = input_data[config.inputs_names.index("Speed")]  # get the speed from input data
        speed = network_manager.fc_block(speed, 128)
        speed = network_manager.fc_block(speed, 128)

    """ Joint sensory """
    j = tf.concat([image_feature, speed], 1)
    j = network_manager.fc_block(j, 512)

    branches_image_delta = make_branches(config.branch_config, image_feature, j, tf, network_manager, "images_refine", branched_features=branches_map)
    # TODO: define the structure and the loss, make sure structure is correct

    out_branches = add_map_and_image_delta_with_threshold(branches_map, branches_image_delta, branches_map_sigma,
                                           config.branch_config, config.branch_config_map, tf)

    weights = network_manager.get_weigths_dict()
    features = network_manager.get_feat_tensors_dict()


    out_branches.append((branches_map, branches_map_sigma))
    return out_branches, None, features, weights, None
