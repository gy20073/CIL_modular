import numpy as np
from codification import *

def single_branch_wp(image_input, speed, control_input, config, sess, train_manager):
    return single_branch(image_input, speed, control_input, config, sess, train_manager, use_wp=True)

def single_branch(image_input, speed, control_input, config, sess, train_manager, use_wp=False):
    branches = train_manager._output_network
    x = train_manager._input_images
    dout = train_manager._dout
    input_speed = train_manager._input_data[config.inputs_names.index("Speed")]
    input_control = train_manager._input_data[config.inputs_names.index("Control")]

    image_input = image_input.reshape((1, config.image_size[0], config.image_size[1], config.image_size[2]))

    speed = np.array(speed / config.speed_factor)

    speed = speed.reshape((1, 1))

    if control_input == 2 or control_input == 0.0:
        all_net = branches[0]
    elif control_input == 3:
        all_net = branches[2]
    elif control_input == 4:
        all_net = branches[3]
    elif control_input == 5:
        all_net = branches[1]

    feedDict = {x: image_input, input_speed: speed, dout: [1] * len(config.dropout)}

    output_all = sess.run(all_net, feed_dict=feedDict)

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

    predicted_speed = sess.run(branches[4], feed_dict=feedDict)
    predicted_speed = predicted_speed[0][0]
    real_speed = speed * config.speed_factor
    print(' REAL PREDICTED ', predicted_speed * config.speed_factor)

    print(' REAL SPEED ', real_speed)
    real_predicted = predicted_speed * config.speed_factor
    if real_speed < 5.0 and real_predicted > 6.0:  # If (Car Stooped) and ( It should not have stoped)
        print('BOOSTING')
        predicted_acc = 1 * (20.0 / config.speed_factor - speed) + predicted_acc  # print "DURATION"

        predicted_brake = 0.0

        predicted_acc = predicted_acc[0][0]

    if use_wp:
        return predicted_steers, predicted_acc, predicted_brake, predicted_wp1_angle, predicted_wp2_angle
    else:
        return predicted_steers, predicted_acc, predicted_brake

def seg_viz(image_input, speed, config, sess, train_manager):
    branches = train_manager._output_network
    x = train_manager._input_images
    dout = train_manager._dout
    input_speed = train_manager._input_data[config.inputs_names.index("Speed")]

    # print ('Image Size Tensor: ', 1,config.image_size[0],config.image_size[1],config.image_size[2])

    image_input = image_input.reshape((1, config.image_size[0], config.image_size[1], config.image_size[2]))

    speed = np.array(speed / config.speed_factor)

    speed = speed.reshape((1, 1))

    gray_images_tensor = train_manager._gray

    feedDict = {x: image_input, input_speed: speed, dout: [1] * len(config.dropout)}

    gray_images = sess.run(gray_images_tensor, feed_dict=feedDict)

    return gray_images

