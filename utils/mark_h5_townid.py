import h5py, glob

if __name__ == "__main__":
    input_id = "exptown_v2_noise10_way"
    townid = 11 # 10 is the rfs_sim town

    base_path = "/scratch/yang/aws_data/carla_collect/"+str(input_id)+"/*/data_*.h5"

    for file in glob.glob(base_path):
        print(file)
        hin = h5py.File(file, 'r+')
        hin["targets"][:, 57] = townid
        hin.close()
