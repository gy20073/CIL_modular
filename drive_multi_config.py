import sys
from drive import drive

sys.path.append('drive_interfaces/configuration')

if __name__ == "__main__":
    driver_config = "9cam_agent_carla_acquire_rc_batch"
    memory = 0.25
    name = "tobefilled_should be summarization of the current config"

    for config in []:
        driver_conf_module = __import__(driver_config)
        driver_conf = driver_conf_module.configDrive()

        drive(args.experiment_name, driver_conf, args.name, float(args.memory))