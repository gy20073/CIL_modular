import h5py, glob, argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='mark some dataset with an id')
    parser.add_argument('-ds', '--dataset', help="input dataset id")
    parser.add_argument('-t', '--townid', default=11, help="which town to mark")
    args = parser.parse_args()

    input_id = args.dataset
    townid = int(args.townid) # 10 is the rfs_sim town

    base_path = "/scratch/yang/aws_data/carla_collect/"+str(input_id)+"/*/data_*.h5"

    for file in glob.glob(base_path):
        print(file)
        hin = h5py.File(file, 'r+')
        hin["targets"][:, 57] = townid
        hin.close()
