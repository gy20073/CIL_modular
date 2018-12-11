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

def make_branches(branch_config, input_feature, input_feature_with_speed, tf, network_manager, scope_name, num_gmm_components):
    branches = []
    """Start BRANCHING"""
    with tf.variable_scope(scope_name):
        for i in range(0, len(branch_config)):
            with tf.name_scope("Branch_" + str(i)):
                if branch_config[i][0] == "Speed":
                    main_feature = input_feature
                else:
                    main_feature = input_feature_with_speed

                branch_output = network_manager.fc_block(main_feature, 256)
                branch_output = network_manager.fc_block(branch_output, 256)

                this_branch = []
                for ivar in range(len(branch_config[i])):
                    gaussians = []
                    # for each variable
                    for icomponent in range(num_gmm_components):
                        mu_sigma = network_manager.fc(branch_output, 2) # the mu and sigma
                        # create a gaussian with these params
                        dist = tf.contrib.distributions.Normal(loc=mu_sigma[:, 0],
                                                               scale=tf.abs(mu_sigma[:, 1])+1e-3,
                                                               validate_args=True,
                                                               allow_nan_stats=False,
                                                               name="normal_dist_branch_" + str(i) +
                                                                    "_var_" + branch_config[i][ivar] +
                                                                    "_component_" + str(icomponent))
                        gaussians.append(dist)
                    # make a mixture distribution out of all these gaussians
                    mixture_coeffi_logits = network_manager.fc(branch_output, num_gmm_components)
                    # the tensor above has shape batchsize*num_gmm_components
                    cat=tf.contrib.distributions.Categorical(logits=mixture_coeffi_logits,
                                                         allow_nan_stats=False,
                                                         name="cat_branch_" + str(i) +
                                                              "_var_" + branch_config[i][ivar])

                    # then create the mixture
                    mixture = tf.contrib.distributions.Mixture(cat, gaussians,
                                                               validate_args=True,
                                                               allow_nan_stats=False,
                                                               name='Mixture' + str(i) +
                                                                    "_var_" + branch_config[i][ivar])
                    this_branch.append(mixture)

                print(this_branch)
                branches.append(this_branch)

    return branches


def create_structure(tf, input_image, input_data, input_size, dropout, config):
    # has a few losses: the ma
    x = input_image
    '''
    mapping = input_data[config.inputs_names.index("mapping")]
    # now have shape batchsize*(50*75)
    mapping = tf.reshape(mapping, [-1, config.map_height, config.map_height * 3 // 2, 1])
    mapping = tf.image.resize_bilinear(mapping, [39, 52], name="resize_mapping")
    '''
    network_manager = Network(config, dropout, tf.shape(x))

    # begin the definition of the real architecture
    image_feature = mapping_tower(input_image, network_manager, tf, "image_tower", channel_multiplier=2)

    """ Speed (measurements)"""
    with tf.name_scope("Speed"):
        speed = input_data[config.inputs_names.index("Speed")]  # get the speed from input data
        speed = network_manager.fc_block(speed, 128)
        speed = network_manager.fc_block(speed, 128)

    """ Joint sensory """
    j = tf.concat([image_feature, speed], 1)
    j = network_manager.fc_block(j, 512)

    branch_GMM = make_branches(config.branch_config, image_feature, j, tf, network_manager, "images_refine", config.GMM_ncomponents)

    weights = network_manager.get_weigths_dict()
    features = network_manager.get_feat_tensors_dict()

    return branch_GMM, None, features, weights, None
