import sys, h5py, cv2, os, shutil, glob
import numpy as np
from subprocess import call

num_images_per_h5 = 200
temp_folder = "./temp/"

def sample_images_from_h5(path, temp):
    f=h5py.File(path, "r")
    if not os.path.exists(temp):
        os.mkdir(temp)

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
        cv2.imwrite(os.path.join(temp, str(imid).zfill(5)+".png"), merged)
    print("done")

if __name__ == "__main__":
    path_pattern = sys.argv[1]

    for path in glob.glob(path_pattern):
        sample_images_from_h5(path, temp_folder)

        call("ffmpeg -y -i " + temp_folder + "%05d.png -c:v libx264 " + path + ".mp4", shell=True)

        shutil.rmtree(temp_folder)
