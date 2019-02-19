import tensorflow as tf
import copy

def mse_branched(network_outputs, ground_truths, control_input, config, **kwargs):
    # typical input: network_outputs: output from network,
    #                ground_truths: _targets_data
    #                control_input: _input_data[self._config.inputs_names.index("Control")]
    '''
    branch_config = [["Steer", "Gas", "Brake"],
                     ["Steer", "Gas", "Brake"],
                     ["Steer", "Gas", "Brake"],
                     ["Steer", "Gas", "Brake"], ["Speed"]]
    '''
    branches_configuration = config.branch_config  # Get the branched configuration

    error_vec = []
    energy_vec = []
    loss_function = 0.0
    for ibranch in range(len(branches_configuration)):
        energy_branch = []
        error_branch = []

        print("network output ", ibranch, network_outputs[ibranch])
        network_outputs_split = tf.split(network_outputs[ibranch], network_outputs[ibranch].get_shape()[1], 1)

        print("branch configuration ", ibranch, branches_configuration[ibranch])
        for i_within_branch in range(len(branches_configuration[ibranch])):
            # Get the name of the target data to be taken
            target_name = branches_configuration[ibranch][i_within_branch]  # name of the target
            target_gt = ground_truths[config.targets_names.index(target_name)]
            # Yang: control branch is in the front, and the latter targets are not branched
            if ibranch < config.inputs_sizes[config.inputs_names.index('Control')]:
                branch_selection = tf.reshape(control_input[:, ibranch], tf.shape(ground_truths[0]))
            else:
                branch_selection = tf.ones(tf.shape(target_gt))

            square_dist = tf.pow(tf.subtract(target_gt, network_outputs_split[i_within_branch]), 2) * branch_selection
            dist = tf.abs(tf.subtract(target_gt, network_outputs_split[i_within_branch])) * branch_selection
            error_branch.append(dist)

            energy_branch.append(square_dist)
            loss_function = loss_function + square_dist * \
                                            config.branch_loss_weight[ibranch] * \
                                            config.variable_weight[target_name]

        energy_vec.append(energy_branch)
        error_vec.append(error_branch)

    if hasattr(config, "weight_decay"):
        wd = config.weight_decay
        if wd > 1e-8:
            print("using weight decay ", wd)
            decay_set = [tf.nn.l2_loss(v) for v in tf.trainable_variables()]
            l2_loss = wd * tf.add_n(decay_set)
            loss_function = loss_function + l2_loss

    '''
    if "dis_to_road_border" in config.inputs_names:
        print(" use the hinge loss of inside road")
        all_inputs=kwargs["all_inputs"]
        dist = all_inputs[config.inputs_names.index('dis_to_road_border')]
        # this is a batch_size * 1 matrix
        loss_hinge = tf.squeeze(tf.maximum(-dist + 0.5, 0.0))
        tf.summary.scalar("loss_inside_road", tf.reduce_mean(loss_hinge))
        loss_function = loss_function + loss_hinge
        # print(loss_function, error_vec, energy_vec)
    '''

    if hasattr(config, "loss_onroad"):
        print("the onroad loss")
        all_inputs = kwargs["all_inputs"]
        onroad = all_inputs[config.inputs_names.index('is_onroad')]
        onroad_logits = network_outputs[-1]
        onroad_gt = tf.cast(onroad, tf.int32)
        loss_onroad = tf.losses.sparse_softmax_cross_entropy(onroad_gt,
                                                             onroad_logits,
                                                             reduction=tf.losses.Reduction.NONE)
        loss_onroad = tf.reshape(loss_onroad, (-1, 1))
        tf.summary.scalar("loss_inside_road", tf.reduce_mean(loss_onroad))
        loss_function = loss_function + loss_onroad * config.loss_onroad

    return loss_function, error_vec, energy_vec, None, branch_selection

def mse_coarse_to_fine(network_outputs, ground_truths, control_input, config):
    branches_refined = network_outputs[:-1]
    branches_map = network_outputs[-1][0]
    branches_map_sigma = network_outputs[-1][1]

    # using this config.branch_config
    config_map = copy.deepcopy(config)
    config_map.branch_config = config.branch_config_map
    loss_refined, error_refined, energy_refined, _, branch_sel_refined = mse_branched(branches_refined, ground_truths, control_input, config)
    # TODO: visualize the errors for the map prediction
    loss_map, error_map, energy_map, _, branch_sel_map = mse_branched(branches_map, ground_truths, control_input, config_map)

    loss_sigma = 0.0
    # TODO check the shape
    for one_branch in branches_map_sigma:
        squared = tf.pow(one_branch, 2)
        sum_over_attributes = tf.reduce_sum(squared, 1, keep_dims=True)
        loss_sigma = loss_sigma + sum_over_attributes


    tf.summary.scalar('loss_refined_manual', tf.reduce_mean(loss_refined))
    tf.summary.scalar('loss_map_manual', tf.reduce_mean(loss_map))
    tf.summary.scalar('loss_sigma_manual', tf.reduce_mean(loss_sigma))

    loss = config.coarse2fine_refined * loss_refined + \
           config.coarse2fine_map * loss_map + \
           config.coarse2fine_sigma * loss_sigma

    print("loss sigma", loss_sigma, "loss refined", loss_refined, "loss map", loss_map)
    #print(loss, error_refined, energy_refined)
    return loss, error_refined, energy_refined, None, None

def gmm(network_outputs, ground_truths, control_input, config):
    # multiple losses for GMM with param phi_i, mu_i, sigma_i
    branches_configuration = config.branch_config  # Get the branched configuration

    loss_function = 0.0
    error_vec = []

    total_log_prob = 0.0
    total_phi_sparsity = 0.0
    total_sigma_normalize = 0.0

    for ibranch in range(len(branches_configuration)):
        for ivar in range(len(branches_configuration[ibranch])):
            target_name = branches_configuration[ibranch][ivar]  # name of the target
            target_gt = ground_truths[config.targets_names.index(target_name)] # this is a batchsize * targetlen tensor

            if ibranch < config.inputs_sizes[config.inputs_names.index('Control')]:
                branch_selection = tf.reshape(control_input[:, ibranch], tf.shape(ground_truths[0]))
            else:
                branch_selection = tf.ones(tf.shape(target_gt))

            mixture = network_outputs[ibranch][ivar]
            # compute all losses
            # - log like
            print("shape of target_gt is ", target_gt)
            loss_log_prob = -mixture.log_prob(tf.squeeze(target_gt))
            # TODO: from here, sanity check
            # sparsity on phi_i
            mixing_coeff = mixture.cat.probs # this should be a batch * num_component tensor
            print("shape of the mixing coeff is ", mixing_coeff)
            loss_phi_sparsity = tf.norm(mixing_coeff, ord=0.5, axis=1)

            # expected scale on log sigma_i
            sigmas = [x.scale for x in mixture.components]
            print("shape of the sigmas", sigmas)
            loss_sigma_normalize = 0.0
            for sigma in sigmas:
                loss_sigma_normalize = loss_sigma_normalize + tf.pow(tf.log(sigma) - config.gmm_sigma_expectation, 2)

            print("shape of the losses", loss_log_prob, loss_sigma_normalize, loss_phi_sparsity, branch_selection)
            # accumulate the loss
            loss_function += (loss_log_prob * config.gmm_w_log_prob +
                              loss_phi_sparsity * config.gmm_w_phi_sparsity +
                              loss_sigma_normalize * config.gmm_w_sigma_normalize) * tf.squeeze(branch_selection)

            total_log_prob += loss_log_prob * tf.squeeze(branch_selection)
            total_phi_sparsity += loss_phi_sparsity * tf.squeeze(branch_selection)
            total_sigma_normalize += loss_sigma_normalize * tf.squeeze(branch_selection)

    tf.summary.scalar('loss_log_prob', tf.reduce_mean(total_log_prob))
    tf.summary.scalar('loss_phi_sparsity', tf.reduce_mean(total_phi_sparsity))
    tf.summary.scalar('loss_sigma_normalize', tf.reduce_mean(total_sigma_normalize))

    # summarize the 3 type of losses
    # print(loss_function, error_vec, energy_vec)
    return loss_function, error_vec, error_vec, None, None


def mse_branched_cls_reg(network_outputs, ground_truths, control_input, config):
    # typical input: network_outputs: output from network,
    #                ground_truths: _targets_data
    #                control_input: _input_data[self._config.inputs_names.index("Control")]
    '''
    branch_config = [["Steer", "Gas", "Brake"],
                     ["Steer", "Gas", "Brake"],
                     ["Steer", "Gas", "Brake"],
                     ["Steer", "Gas", "Brake"], ["Speed"]]
    '''
    branches_configuration = config.branch_config  # Get the branched configuration

    error_vec = []
    energy_vec = []
    loss_function = 0.0

    predicted_label_vec = []
    gt_label_vec = []

    if hasattr(config, "classification_weighting"):
        weighting = tf.constant(
            config.classification_weighting,
            dtype=tf.float32,
            shape=(len(config.classification_weighting), ),
            name='const_weighting')
    else:
        weighting = None

    for ibranch in range(len(branches_configuration)):
        energy_branch = []
        error_branch = []

        print("network output ", ibranch, network_outputs[ibranch])
        cls_logits = network_outputs[ibranch][0]
        # here is the one of k encoding of the ground truth
        target_gt = ground_truths[config.targets_names.index('Waypoint_Shape')]
        target_gt = tf.cast(target_gt, tf.int32)
        target_gt = tf.reshape(target_gt, (-1, ))

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_gt, logits=cls_logits, name="waypoint_classification")

        # weighted loss on different classes
        if weighting is not None:
            weights = tf.gather(weighting, target_gt, axis=0)
            print("weights ", weights)
            losses = losses * weights
            print("losses ", losses)

        losses = tf.expand_dims(losses, axis=-1)
        branch_selection = tf.reshape(control_input[:, ibranch], tf.shape(ground_truths[0]))
        error_branch.append(losses*branch_selection)
        energy_branch.append(losses*branch_selection)

        loss_function += tf.reduce_mean(losses*branch_selection) * \
                         config.branch_loss_weight[ibranch] * \
                         config.variable_weight['Waypoint_Shape']
        # here goes the classification loss
        # then we want to summarize the accuracy
        predicted_label = tf.argmax(cls_logits, axis=-1, output_type=tf.int32)
        branch_selection_bool = tf.cast(branch_selection, tf.bool)

        predicted_label_vec.append(tf.boolean_mask(tf.squeeze(predicted_label), tf.squeeze(branch_selection_bool)))
        gt_label_vec.append(tf.boolean_mask(tf.squeeze(target_gt), tf.squeeze(branch_selection_bool)))

        correct = tf.equal(predicted_label_vec[-1], gt_label_vec[-1])
        correct = tf.cast(correct, tf.float32)
        tf.summary.scalar('accuracy_%d' % ibranch, tf.reduce_mean(correct))

        network_outputs_split = [None] + tf.split(network_outputs[ibranch][1], network_outputs[ibranch][1].get_shape()[1], 1)

        print("branch configuration ", ibranch, branches_configuration[ibranch])
        for i_within_branch in range(1, len(branches_configuration[ibranch])):
            # Get the name of the target data to be taken
            target_name = branches_configuration[ibranch][i_within_branch]  # name of the target
            target_gt = ground_truths[config.targets_names.index(target_name)]
            # Yang: control branch is in the front, and the latter targets are not branched
            if ibranch < config.inputs_sizes[config.inputs_names.index('Control')]:
                branch_selection = tf.reshape(control_input[:, ibranch], tf.shape(ground_truths[0]))
            else:
                branch_selection = tf.ones(tf.shape(target_gt))

            square_dist = tf.pow(tf.subtract(target_gt, network_outputs_split[i_within_branch]), 2) * branch_selection
            if hasattr(config, "mse_self_normalize") and config.mse_self_normalize:
                print("normalizing with target data")
                square_dist /= (tf.pow(target_gt, 2) + 0.01)

            dist = tf.abs(tf.subtract(target_gt, network_outputs_split[i_within_branch])) * branch_selection
            error_branch.append(dist)

            energy_branch.append(square_dist)
            loss_function = loss_function + square_dist * \
                                            config.branch_loss_weight[ibranch] * \
                                            config.variable_weight[target_name]

        energy_vec.append(energy_branch)
        error_vec.append(error_branch)

    # the accuracies summarization
    predicted_label_vec = tf.concat(predicted_label_vec, axis=0)
    gt_label_vec = tf.concat(gt_label_vec, axis=0)
    correct = tf.equal(predicted_label_vec, gt_label_vec)
    correct = tf.cast(correct, tf.float32)
    tf.summary.scalar('accuracy_overall', tf.reduce_mean(correct))

    # compute the confusion matrix
    # Compute a per-batch confusion
    batch_confusion = tf.confusion_matrix(gt_label_vec, predicted_label_vec,
                                          num_classes=config.waypoint_num_shapes,
                                          name='batch_confusion')
    # Create an accumulator variable to hold the counts
    confusion = tf.Variable(tf.zeros([config.waypoint_num_shapes, config.waypoint_num_shapes],
                                     dtype=tf.int32), name='confusion')
    # Create the update op for doing a "+=" accumulation on the batch
    confusion_update = confusion.assign(confusion + batch_confusion)
    # Cast counts to float so tf.summary.image renormalizes to [0,255]
    confusion_image = tf.reshape(tf.cast(confusion, tf.float32),
                                 [1, config.waypoint_num_shapes, config.waypoint_num_shapes, 1])
    tf.summary.image('confusion', confusion_image)

    with tf.get_default_graph().control_dependencies([confusion_update]):
        loss_function = tf.identity(loss_function)

    if hasattr(config, "weight_decay"):
        wd = config.weight_decay
        if wd > 1e-8:
            print("using weight decay ", wd)
            decay_set = [tf.nn.l2_loss(v) for v in tf.trainable_variables()]
            l2_loss = wd * tf.add_n(decay_set)
            loss_function = loss_function + l2_loss

    return loss_function, error_vec, energy_vec, None, branch_selection
