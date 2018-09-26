class configDrive:
    # The config_driver is related to carla driving stuff. All outside of the Game configuration must be placed here
    # TODO: kind of change this to be CarlaSettings Based ?
    def __init__(self):
        # self.experiment_name =''
        self.carla_config = "./drive_interfaces/carla/yang_template_3cams_103.ini" # TODO: randomize the camera setting
        # TODO: figure out how to collect human demo under multiple weather conditions

        self.host = "127.0.0.1"
        self.port = 2000
        self.path = "/Users/yang/Downloads/vladlen/human_data/"  # If path is set go for it , if not expect a name set
        self.resolution = [768, 576]
        self.noise = "None"  # NON CARLA SETTINGS PARAM # TODO: experiment with this
        self.type_of_driver = "Human"
        self.interface = "Carla"

        self.image_cut = [0, None]  # This is made from top to botton # TODO: finalize this
        self.autopilot = False # This should not be related in the carla machine, but might matter in CarlaHuman?
        self.reset_period = 960
        # Figure out a solution for setting specific properties of each interface
        # TODO: if not using planner, has to press keyboards, otherwise has to have the end goal with a planner
        self.use_planner = True # we want to get planing instruction from the planner, and human follow the rule

        self.num_images_to_collect = 200 * 20  # how many images to collect in total
        self.re_entry = True  # True to allow continue collecting the data, this will make changes to the folder structure

        self.city_name = 'Town01'

        # Test parameters to be shared between models

        self.timeouts = [200.0]  # 130
        self.weather = "1"  # TODO: randomize the weather
        self.cars = "50"
        self.pedestrians = "100" # TODO: there is no car and pedestrain support in 0.9.0


        # TODO: figure out the recording capability
        self.show_screen = True  # NON CARLA SETTINGS PARAM
        self.aspect_ratio = [3, 1]  # only matters when showing the screen
        self.scale_factor = 1  # NON CARLA SETTINGS PARAM
        self.plot_vbp = False
        self.number_screens = 1 # added later

        self.carla_api_version = "0.9.0"