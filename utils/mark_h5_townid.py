import h5py, glob

if __name__ == "__main__":
    input_id = "steer103_v5_way_v2"
    townid = 1

    base_path = "/data/yang/code/aws/scratch/carla_collect/"+str(input_id)+"/*/data_*.h5"

    for file in glob.glob(base_path):
        print(file)
        hin = h5py.File(file, 'r+')
        hin["targets"][:, 57] = townid
        hin.close()
