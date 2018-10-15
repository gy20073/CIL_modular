import h5py, glob, random, pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

path = "/scratch/yang/aws_data/carla_collect/steer103_v4_waypoint/*/*h5"
threshold_stop = 0.5
ncluster_class = 30
center_name = "cluster_centers.npy.v4"
# below won't change
index_id = 55
index_scale = 56

def plot_wp(wps):
    plt.plot(wps[:, 0], wps[:, 1], 'x')
    plt.xlim(0, 1)
    plt.ylim(-1, 1)
    plt.show()

def plot_wp_noscale(wps):
    plt.plot(wps[:, 0], wps[:, 1], 'x')
    plt.show()

all_wps = []
for f in sorted(glob.glob(path)):
    h5 = h5py.File(f, "r")
    targets = h5["targets"]
    wps = targets[:, 35:(35+int(targets[0, -1]))]
    wps = np.reshape(wps, (200, -1, 2))
    all_wps.append(wps)
    h5.close()

wps = np.concatenate(all_wps, axis=0)
original_wps = wps

non_stop = wps[:,-1,0]>threshold_stop
wps = wps[non_stop, :, :]
scale0 = np.abs(wps[:, -1:, 0:1])

wps[:, :, 0:1] /= scale0
# decide to keep the aspect ratio and use the same constant to normalize the trajectory
wps[:, :, 1:2] /= scale0

print(np.max(wps[:,:,0]))
print(np.min(wps[:,:,0]))
print(np.max(wps[:,:,1]))
print(np.min(wps[:,:,1]))

# begin the clustering process
data = np.reshape(wps, (wps.shape[0], -1))

kmeans = KMeans(n_clusters=ncluster_class, n_jobs=-1, verbose=2, random_state=1)
kmeans.fit(data)
print(kmeans.cluster_centers_)

# save the centers
pickle.dump(kmeans.cluster_centers_, open(center_name, "wb"))

# compute the id for each of the datapoints, and the scale
id = np.zeros((original_wps.shape[0], ))
id[:] = ncluster_class # this is the stop classes
id[non_stop] = kmeans.labels_

scale = np.zeros((original_wps.shape[0], ))
scale[non_stop] = np.reshape(scale0, (-1, ))

# finally write the cluster id and the scale into the h5 file
i = 0
for f in sorted(glob.glob(path)):
    print(f)
    h5 = h5py.File(f, "r+")
    h5["targets"][:, index_id] = id[i*200:(i+1)*200]
    h5["targets"][:, index_scale] = scale[i * 200:(i + 1) * 200]
    h5.close()
    i += 1

