import sys, os, time, threading

TownName = "Town05"
start_port=2200
available_gpus = [0]
num_processes = 1
use_docker = False
# 9cam_agent_carla_acquire_rc_batch_090, change its contents

if TownName == "Exp_Town" or TownName == "Town05":
    CARLA_PATH = "/scratch/yang/aws_data/carla_095/CarlaUE4.sh"
    os.environ['CARLA_VERSION'] = '0.9.5'
    docker_path = None
    town_within_path = TownName
elif TownName == "Town03" or TownName == "Town04" or TownName == "RFS_MAP":
    CARLA_PATH = "/scratch/yang/aws_data/carla_auto2/CarlaUE4.sh"
    os.environ['CARLA_VERSION'] = '0.9.auto2'
    docker_path = "gy20073/carla_auto2:latest"
    town_within_path = "/Game/Carla/Maps/" + TownName
else:
    CARLA_PATH = "/scratch/yang/aws_data/carla_0.8.4/CarlaUE4.sh"
    docker_path = "gy20073/carla_084:latest"
    town_within_path = "/Game/Maps/" + TownName


from configparser import ConfigParser
from drive import drive
from multiprocessing import Process

sys.path.append('drive_interfaces/configuration')


def config_naming(tag, config, weather):
    out = tag + "_" + \
          config[0] + "=" + config[1] + "_" + \
          "WeatherId=" + str(weather).zfill(2)
    return out


def config_change_attrs(src, dst, new_attrs):
    config = ConfigParser()
    config.optionxform = str
    config.read(src)
    for one in new_attrs:
        config.set(one[0], one[1], one[2])
    with open(dst, "w") as f:
        config.write(f)

def process_collect(list_of_configs, port, gpu,
                    tag, generated_config_cache_path, template_path, driver_config, TownName):
    print("port is ", port, "!!!!!!!!!!!!!!!!!!")
    port = int(port)
    count = 5  # to flag that initially we need to start the server
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for config, weather in list_of_configs:
        # generate this config
        this_name = config_naming(tag, config, weather)
        config_fname = os.path.join(generated_config_cache_path, this_name + ".ini")

        config_change_attrs(template_path, config_fname,
                            [("CARLA/Sensor", config[0], config[1]),
                             ("CARLA/LevelSettings", "WeatherId", str(weather))])

        driver_conf_module = __import__(driver_config)
        driver_conf = driver_conf_module.configDrive()
        driver_conf.carla_config = config_fname
        driver_conf.weather = str(weather)
        driver_conf.port = port

        # experiment_name & memory not used for human

        while True:
            count += 1
            if count >= 5:
                count = 0
                if use_docker:
                    cmd = ["docker run -p %d-%d:%d-%d --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=%d %s /bin/bash CarlaUE4.sh %s -carla-server -benchmark -fps=5 -carla-world-port=%d" % (port, port+2, port, port+2, gpu, docker_path, town_within_path, port)]
                else:
                    cmd = ['bash', '-c', " '%s %s  -carla-server -benchmark -fps=5 -carla-world-port=%d' " % (CARLA_PATH, town_within_path, port)]

                print(" ".join(cmd))
                print("before spawnling")
                t = threading.Thread(target=lambda: os.system(" ".join(cmd)))
                t.start()

                time.sleep(60)

            if drive("", driver_conf, this_name, 0):
                count = 0
                break
            time.sleep(1)

        print("finished one setting, sleep for 3 seconds")
        time.sleep(3)


if __name__ == "__main__":
    driver_config = "9cam_agent_carla_acquire_rc_batch_095"

    generated_config_cache_path = "./drive_interfaces/carla/auto_gen_configs/"
    tag = "default"
    # TODO: tune those base config
    template_path = "./drive_interfaces/carla/yang_template_3cams_103.ini"

    # TODO: check whether those are implemented to all 3 cameras
    # the noiser setting
    configs = [("RotationPitch", "0"),  # This is the default one
               ("RotationPitch", "5"),
               ("RotationPitch", "-5"),
               ("ImageSizeX", "700"),
               ("ImageSizeX", "800"),
               ("PositionZ", "1.4"),
               ("PositionZ", "1.8")]
    weather_range = range(1, 15)

    # in total there are 7*14 = 100 configs, each of the 200 h5 file has size of 33M, i.e. 30 h5 = 1G
    # Thus we aim to collect 100 hours of training, that is 400G, so each config has quota of 3G, which is 100 files
    # an initial config ends here

    #available_gpus = [0, 2, 4, 5, 6]
    #num_processes = len(available_gpus) * 2

    list_of_configs = [[] for i in range(num_processes)]

    counter = 0
    for config in configs:
        for weather in weather_range:
            list_of_configs[counter % num_processes].append((config, weather))
            counter += 1

    ps=[]
    for i in range(num_processes):
        p = Process(target=process_collect, args=(list_of_configs[i],
                                                  start_port+i*3,
                                                  available_gpus[i % len(available_gpus)],
                                                  tag,
                                                  generated_config_cache_path,
                                                  template_path,
                                                  driver_config,
                                                  TownName))
        p.start()
        print("after starts!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        time.sleep(3)
        ps.append(p)
        print("finsished starting process ", i)

    for i in range(num_processes):
        ps[i].join()
