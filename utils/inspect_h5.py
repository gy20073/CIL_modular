import sys, h5py, cv2, os, shutil, glob, math
import numpy as np
from subprocess import call
from PIL import Image, ImageDraw, ImageFont

num_images_per_h5 = 200
temp_folder = "./temp/"


def write_text_on_image(image, string, fontsize=10):
    image = image.copy()
    image = np.uint8(image)
    j = Image.fromarray(image)
    draw = ImageDraw.Draw(j)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", fontsize)
    draw.text((0, 0), string, (255, 0, 0), font=font)

    return np.array(j)

def plot_waypoints_on_image(image, wps, scale_factor, dot_size):
    imsize = image.shape
    wps = np.concatenate(([[0,0]], wps), axis=0)
    for i in range(wps.shape[0]):
        wp = wps[i]

        depth = wp[0] + 2.46 - 0.7 + 2.0 # TODO
        horizontal = wp[1]
        vertical = -1.6
        h, v = point_to_2d(depth, horizontal, vertical)

        xoff = int((-v + 0.5) * imsize[0])
        yoff = int((h + 0.5) * imsize[1])
        image[xoff - dot_size: xoff + dot_size, yoff - dot_size: yoff + dot_size, 0] = 0
        image[xoff - dot_size: xoff + dot_size, yoff - dot_size: yoff + dot_size, 1] = 0
        image[xoff - dot_size: xoff + dot_size, yoff - dot_size: yoff + dot_size, 2] = 255

    return image

    # TODO plotting the waypoints on the ground

def point_to_2d(depth, horizontal, vertical, half_width_fov=math.radians(103.0)/2, half_height_fov = math.atan(0.75*math.tan(math.radians(103.0)/2))):
    # the horizontal and vertical are both relative to the center, same as the output
    h = horizontal / depth * 0.5 / math.tan(half_width_fov)
    v = vertical   / depth * 0.5 / math.tan(half_height_fov)
    return h, v


def sample_images_from_h5(path, temp, show_all, is3, pure_video):
    f=h5py.File(path, "r")
    if not os.path.exists(temp):
        os.mkdir(temp)

    if show_all:
        images = {}
        print("reading images from h5")
        for key in ['CameraLeft', 'CameraMiddle', 'CameraRight']:
            images[key] = []
            for imid in range(num_images_per_h5):
                data = f[key][imid]
                img = cv2.imdecode(data, 1)
                images[key].append(img)
        print("concating and writing out images")
        for imid in range(num_images_per_h5):
            l = []
            for key in ['CameraLeft', 'CameraMiddle', 'CameraRight']:
                l.append(images[key][imid])
            merged = np.concatenate(l, axis=1)
            cv2.imwrite(os.path.join(temp, str(imid).zfill(5)+".jpg"), merged)

    else:
        image_list = [None, None, None]
        counter = 0
        key = 'CameraMiddle'
        for imid in range(f[key].shape[0]):
            path = os.path.join(temp, str(counter).zfill(5)+".jpg")

            image = cv2.imdecode(f[key][imid], 1)

            if image.shape[0] > 300:
                image = image[::2, ::2, :]
            if image.shape[0] < 100:
                image = cv2.resize(image, (image.shape[1]*2, image.shape[0]*2))

            #image = image[:,:,::-1]
            td = lambda fl: "{:.2f}".format(fl)
            font = int(np.ceil(25.0 / (576/2) * image.shape[0])) + 1
            if not pure_video:
                image = write_text_on_image(image,
                                            "steer    :" + td(f["targets"][imid, 0]) + "\n" +
                                            "throttle :" + str(f["targets"][imid, 1]) + "\n" +
                                            "brake    :" + str(f["targets"][imid, 2]) + "\n" +
                                            "direction:" + str(f["targets"][imid, 24]) + "\n" +
                                            "speed    :" + td(f["targets"][imid, 10]) + "\n" +
                                            "ori      :" + td(f["targets"][imid, 21]) + " " + td(f["targets"][imid, 22]) + " " + td(f["targets"][imid, 23]) + "\n" +
                                            "wp1_angle:" + td(f["targets"][imid, 31]) + "\n" ,
                                            fontsize=font)

                # plotting the waypoints on the image
                if f["targets"].shape[1] > 40:
                    # then we have the waypoints stored
                    size = int(f["targets"][imid, 99])
                    flattend = f["targets"][imid, 35:(35+size)]
                    wp = np.reshape(flattend, (-1, 2))
                    image = plot_waypoints_on_image(image, wp, 20, 4)

            if not is3:
                #image = image[:,:,::-1]
                cv2.imwrite(path, image)
                counter += 1
            else:
                mod = imid % 3
                # 0 middle
                # 1 left
                # 2 right
                mapping = {0: 1, 1: 0, 2: 2}
                image_list[mapping[mod]] = image
                if mod == 2:
                    if pure_video:
                        image = image_list[1]
                    else:
                        image = np.concatenate(image_list, axis=1)
                    cv2.imwrite(path, image)
                    counter += 1

    print("done")

if __name__ == "__main__":
    path_pattern = sys.argv[1]

    if len(sys.argv) >= 3:
        is3 = (sys.argv[2].lower().strip() == "true")
    else:
        is3 = False
        print("is3 False")

    if len(sys.argv) >= 4:
        write_all = (sys.argv[3].lower().strip() == "true")
    else:
        write_all = False

    if len(sys.argv) >= 5:
        pure_video = (sys.argv[4].lower().strip() == "true")
    else:
        pure_video = False

    for path in glob.glob(path_pattern):
        try:
            sample_images_from_h5(path, temp_folder, write_all, is3, pure_video)
        except:
            print("Failed to extract from ", path)
            if False:
                # moving the error file to some kind of garbage bin
                head, tail = os.path.split(path)
                garbage_path = os.path.join(head, "bad_h5")
                if not os.path.exists(garbage_path):
                    os.makedirs(garbage_path)
                target_path = os.path.join(garbage_path, tail)
                shutil.move(path, target_path)

                if os.path.exists(temp_folder):
                    shutil.rmtree(temp_folder)
                continue
            else:
                print("warning not moving the file to the garbage bin")

        call("ffmpeg -y -i " + temp_folder + "%05d.jpg -c:v libx264 " + path + ".mp4", shell=True)

        shutil.rmtree(temp_folder)
