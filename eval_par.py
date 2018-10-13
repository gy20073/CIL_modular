from subprocess import Popen
from time import sleep
import math

if __name__ == "__main__":
    gpus_agent = [5]
    gpus_carla = [0]
    gpus_perception = [5]
    num_perception = 1
    exp_id = "mm45_v4_base_newseg_noiser_TL_lane_structure02_goodsteer_waypoint_zoom_eqweight"
    weather_batch_size = 14
    test_name = "YangExp3cam"
    only_town_01 = True
    #test_name = "YangExp3camFov90"
    #test_name = "YangExp3camGTA"
    # TODO make a new test setting to mimic the camera locations
    # num par = 14/3 * 2


    processes = []
    ithread = 0
    for town in ["Town01", "Town02"]:
        next_weather = 1
        for _ in range(int(math.ceil(14.0 / weather_batch_size))):
            weather_id = ""
            for i in range(weather_batch_size):
                if next_weather > 14:
                    break
                weather_id += str(next_weather) + ","
                next_weather += 1
            weather_id = weather_id[:-1]

            percep = ""
            for i in range(num_perception):
                id = ithread*num_perception+i
                percep += str(gpus_perception[id % len(gpus_perception)]) + ","
            percep = percep[:-1]


            cmd = ["/data1/yang/code/aws/CIL_modular/eval_one.sh",
                   str(gpus_agent[ithread % len(gpus_agent)]),
                   str(gpus_carla[ithread % len(gpus_carla)]),
                   percep,
                   weather_id,
                   exp_id,
                   town,
                   test_name]

            # TODO: call the eval once code
            print(cmd)
            p=Popen(cmd)
            processes.append(p)
            sleep(10)

            ithread += 1
        if only_town_01:
            break

    for p in processes:
        p.wait()
