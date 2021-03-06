import numpy as np
from codification import *

def single_branch_wp(image_input, speed, control_input, config, sess, train_manager):
    return single_branch(image_input, speed, control_input, config, sess, train_manager, use_wp=True)

def single_branch(image_input, speed, control_input, config, sess, train_manager, use_wp=False, map=None):
    branches = train_manager._output_network

    control_to_branch = {2:0, 0:0, 3:2, 4:3, 5:1}
    all_net = branches[control_to_branch[int(control_input)]]

    image_input = image_input.reshape((1, config.feature_input_size[0], config.feature_input_size[1], config.feature_input_size[2]))
    speed = np.array(speed / config.speed_factor)
    speed = speed.reshape((1, 1))
    feedDict = {train_manager._input_images: image_input,
                train_manager._input_data[config.inputs_names.index("Speed")]: speed,
                train_manager._dout: [1] * len(config.dropout)}
    if map is not None:
        feedDict.update({train_manager._input_data[config.inputs_names.index("mapping")]: map})

    output_all, predicted_speed = sess.run([all_net, branches[4]], feed_dict=feedDict)

    if use_wp:
        predicted_wp1_angle = (output_all[0][0])
        predicted_wp2_angle = (output_all[0][1])
        predicted_steers = (output_all[0][2])
        predicted_acc = (output_all[0][3])
        predicted_brake = (output_all[0][4])
    else:
        predicted_steers = (output_all[0][0])
        predicted_acc = (output_all[0][1])
        predicted_brake = (output_all[0][2])

    predicted_speed = predicted_speed[0][0]
    real_speed = speed * config.speed_factor
    real_predicted = predicted_speed * config.speed_factor
    print('PREDICTED SPEED ', real_predicted, '\nREAL SPEED ', real_speed)

    if real_speed < 5.0 and real_predicted > 6.0:  # If (Car Stooped) and ( It should not have stoped)
        print('BOOSTING')
        predicted_acc += 20.0 / config.speed_factor - speed  # print "DURATION"
        predicted_brake = 0.0

    if use_wp:
        return predicted_steers, predicted_acc, predicted_brake, predicted_wp1_angle, predicted_wp2_angle
    else:
        return predicted_steers, predicted_acc, predicted_brake


def single_branch_yang_wp(image_input, speed, control_input, config, sess, train_manager, map=None):
    # map is the numpy array outputed by the mapping helper
    branches = train_manager._output_network

    control_to_branch = {2:0, 0:0, 3:2, 4:3, 5:1}
    all_net = branches[control_to_branch[int(control_input)]]

    image_input = image_input.reshape((1, config.feature_input_size[0], config.feature_input_size[1], config.feature_input_size[2]))
    speed = np.array(speed / config.speed_factor)
    speed = speed.reshape((1, 1))
    feedDict = {train_manager._input_images: image_input,
                train_manager._input_data[config.inputs_names.index("Speed")]: speed,
                train_manager._dout: [1] * len(config.dropout)}
    if map is not None:
        feedDict.update({train_manager._input_data[config.inputs_names.index("mapping")]: map})

    output_all, predicted_speed = sess.run([all_net, branches[4]], feed_dict=feedDict)

    waypoints = np.reshape(output_all[0], (-1, 2))

    predicted_speed = predicted_speed[0][0]
    real_speed = speed * config.speed_factor
    real_predicted = predicted_speed * config.speed_factor
    print('PREDICTED SPEED ', real_predicted, '\nREAL SPEED ', real_speed)

    if real_speed < 5.0 and real_predicted > 6.0:  # If (Car Stooped) and ( It should not have stoped)
        print('BOOSTING')

    return waypoints, real_predicted

def single_branch_yang_wp_stack(image_input, speed, control_input, config, sess, train_manager, map=None):
    branches = train_manager._output_network

    control_to_branch = {2:0, 0:0, 3:2, 4:3, 5:1}
    all_net = branches[control_to_branch[int(control_input)]]

    image_input = image_input.reshape((1, config.feature_input_size[0], config.feature_input_size[1], config.feature_input_size[2]))
    speed = np.array(speed / config.speed_factor)
    speed = speed.reshape((1, 1))
    feedDict = {train_manager._input_images: image_input,
                train_manager._input_data[config.inputs_names.index("Speed")]: speed,
                train_manager._dout: [1] * len(config.dropout)}
    if map is not None:
        feedDict.update({train_manager._input_data[config.inputs_names.index("mapping")]: map})

    output_all, predicted_speed, onroad_output = sess.run([all_net, branches[4], branches[-1]], feed_dict=feedDict)

    waypoints = np.reshape(output_all[0][3:], (-1, 2))
    predicted_steers = (output_all[0][0])
    predicted_acc = (output_all[0][1])
    predicted_brake = (output_all[0][2])

    predicted_speed = predicted_speed[0][0]
    real_speed = speed * config.speed_factor
    real_predicted = predicted_speed * config.speed_factor
    print('PREDICTED SPEED ', real_predicted, '\nREAL SPEED ', real_speed)

    if real_speed < 5.0 and real_predicted > 6.0:  # If (Car Stooped) and ( It should not have stoped)
        print('BOOSTING')
        predicted_acc += 20.0 / config.speed_factor - speed  # print "DURATION"
        predicted_brake = 0.0

    return waypoints, predicted_steers, predicted_acc, predicted_brake, real_predicted, onroad_output


def single_branch_yang_cls_reg(image_input, speed, control_input, config, sess, train_manager):
    # the machine output function for classification + scale regression
    branches = train_manager._output_network

    control_to_branch = {2: 0, 0: 0, 3: 2, 4: 3, 5: 1}
    all_net = branches[control_to_branch[int(control_input)]]

    image_input = image_input.reshape(
        (1, config.feature_input_size[0], config.feature_input_size[1], config.feature_input_size[2]))
    speed = np.array(speed / config.speed_factor)
    speed = speed.reshape((1, 1))
    feedDict = {train_manager._input_images: image_input,
                train_manager._input_data[config.inputs_names.index("Speed")]: speed,
                train_manager._dout: [1] * len(config.dropout)}

    output_all = sess.run(all_net, feed_dict=feedDict)

    logits = output_all[0]
    logits = np.squeeze(logits)
    shape_id = np.argmax(logits)
    scale = output_all[1]

    return shape_id, scale


def seg_viz(image_input, speed, config, sess, train_manager):
    image_input = image_input.reshape((1, config.feature_input_size[0], config.feature_input_size[1], config.feature_input_size[2]))

    speed = np.array(speed / config.speed_factor)
    speed = speed.reshape((1, 1))

    feedDict = {train_manager._input_images: image_input,
                train_manager._input_data[config.inputs_names.index("Speed")]: speed,
                train_manager._dout: [1] * len(config.dropout)}

    gray_images = sess.run(train_manager._gray, feed_dict=feedDict)

    return gray_images

