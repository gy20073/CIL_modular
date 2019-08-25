import pickle, sys, cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.append("../utils/")
import mapping_helper

def get_lat_lng_std(sol):
    sp = sol.split("WGS84")
    current = sp[1].strip()
    sp = current.split(" ")
    lat_std = float(sp[0])
    lon_std = float(sp[1])
    return lat_std, lon_std

def loop_over_video(path, func, temp_down_factor=1, batch_size=1, output_name="output.avi", pickle_name=None):
    # from a video, use cv2 to read each frame

    # reading from a video
    cap = cv2.VideoCapture(path)
    with open(pickle_name, "rb") as fin:
        extra_info = pickle.load(fin)

    i = 0
    batch_frames = []
    video_init = False
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            print("breaking")
            break

        if i % temp_down_factor:
            i += 1
            continue
        if i % 10 == 0:
            print(i)
        batch_frames.append(frame)
        if len(batch_frames) == batch_size:
            # frame is the one
            frame_seq = func(batch_frames, extra_info[i])
            if not video_init:
                fourcc = cv2.VideoWriter_fourcc(*'DIVX')
                video = cv2.VideoWriter(output_name, fourcc, 30 // temp_down_factor,
                                        (frame_seq[0].shape[1], frame_seq[0].shape[0]))
                print("in test_video.loop_over_video, loop function output size:", frame_seq[0].shape)
                video_init = True
            for frame in frame_seq:
                video.write(frame)
            batch_frames = []
        i += 1

    cap.release()
    video.release()

def write_text_on_image(image, string, fontsize=10):
    # image = image.copy()
    image = np.uint8(image)
    j = Image.fromarray(image)
    draw = ImageDraw.Draw(j)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", fontsize)
    draw.text((0, 0), string, (255, 0, 0), font=font)

    return np.array(j)

def callback(frames, extra_info, spatial_downsample=2):
    frame = frames[0]

    # split into 3 cams
    H, W, C = frame.shape
    assert (W % 3 == 0)
    sensors = [frame[:, 0:W // 3, :], frame[:, W // 3:W * 2 // 3, :], frame[:, W * 2 // 3:, :]]
    # downsample the images by a factor
    for i in range(3):
        sensors[i] = sensors[i][::spatial_downsample, ::spatial_downsample, :]
    nh = H // spatial_downsample
    nw = W // 3 // spatial_downsample

    speed_ms, condition, pos, ori, dgps, NMEA_tuple = extra_info

    lat_std, lon_std = get_lat_lng_std(NMEA_tuple[-1])
    info_str = "\n" * 12 + \
               "speed: {:.2f} m/s \n".format(speed_ms) + \
               "vehicle heading: {:.2f} rad\n".format(ori) + \
               "lat:{:13.8f} std={:4.1f}cm \n".format(dgps[0], lat_std * 100) + \
               "lng:{:13.8f} std={:4.1f}cm \n".format(dgps[1], lon_std * 100)

    this_map = MH.get_map("rfs", dgps, ori)
    this_map = MH.map_to_debug_image(this_map)[:,:,::-1]

    out = np.zeros((nh * 2, nw * 3, 3), dtype=np.uint8)
    out[:nh, :, :] = np.concatenate(sensors, axis=1)
    out[nh:, nw:nw * 2, :] = cv2.resize(this_map, (nw, nh))
    out = write_text_on_image(out, info_str, fontsize=24)

    return [out]

if __name__ == "__main__":
    # input path begin
    base = "/home/yang/data/cheat_data/2019-08-23_23-39-53"
    # input path end

    video = base + "/video.avi"
    pkl = base + "/video.pkl"

    with open(pkl, "r") as f:
        info = pickle.load(f)

    MH = mapping_helper.mapping_helper(
        output_height_pix=200,
        version="v1")

    loop_over_video(video,
                    callback,
                    temp_down_factor=1,
                    batch_size=1,
                    output_name=base + "/visualization.avi",
                    pickle_name=pkl)
