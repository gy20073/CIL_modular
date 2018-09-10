import json, sys, os
import numpy as np
import glob
import csv



if __name__ == "__main__":
    exp_id = sys.argv[1]

    for town in ["Town01", "Town02"]:
        print(town)
        path = "/scratch/yang/aws_data/CIL_modular_data/_benchmarks_results/" + exp_id + "_*" + "_YangExp3cam_" + town + "/summary.csv"

        results = {}
        for item in glob.glob(path):
            sp = item.split(exp_id)
            now = sp[1]
            sp = now.split("YangExp3cam")
            now = sp[0][1:-1]
            if "_" in now:
                print("ignoring ", item)
                continue

            with open(item) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    line_count += 1
                    if line_count == 1:
                        continue

                    weather = int(row[0])
                    success = int(row[2])
                    if weather in results:
                        results[weather].append(success)
                    else:
                        results[weather] = [success]

        all_train = []
        all_val = []
        for key in sorted(results.keys()):
            if key in range(1, 15, 3):
                phase = "validation"
                all_val.append(results[key])
            else:
                phase = "training  "
                all_train.append(results[key])
            print("phase %s, weather %d, success rate %f, num sample %d" % (phase, key, np.mean(results[key]), len(results[key])) )

        all_train = np.concatenate(all_train)
        print("training mean success rate %f, num sample %d" % (np.mean(all_train), len(all_train)))
        all_val = np.concatenate(all_val)
        print("validation mean success rate %f, num sample %d" % (np.mean(all_val), len(all_val)))

