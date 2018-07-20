
def parse_drive_arguments(args, driver_conf, attributes):
    for attr in attributes:
        value = getattr(args, attr)
        if value is not None:
            setattr(driver_conf, attr, value)

    if args.port is not None:
        driver_conf.port = int(args.port)

    if args.resolution is not None:
        res_string = args.resolution.split(',')
        driver_conf.resolution = [int(res_string[0]), int(res_string[1])]

    if args.image_cut is not None:
        cut_string = args.image_cut.split(',')
        driver_conf.image_cut = [int(cut_string[0]), int(cut_string[1])]

    return driver_conf

import cv2
def arr_to_png(arr):
    return cv2.imencode(".png", arr)