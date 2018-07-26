class configDrive:
    def __init__(self):
        # human or machine
        self.type_of_driver = "Human"

        # resource related
        self.host = "127.0.0.1"
        self.port = 2000
        self.path = "/scratch/yang/aws_data/carla_collect/1/"  # If path is set go for it , if not expect a name set

        # data collection related
        self.city_name = 'Town01'
        self.carla_config = None # This will be filled by the caller
        self.middle_camera = 1  # might be wrong, based on the actual camera? # This is not used in CarlaHuman
        # collect method
        self.autopilot = True
        self.use_planner = True
        self.noise = "Spike"  # NON CARLA SETTINGS PARAM
        self.reset_period = 960
        # Those parameters will override carla_config
        self.weather = "1" # This will be override by the caller
        self.cars = "50"
        self.pedestrians = "100"

        # post processing
        self.image_cut = [0, None]  # This is made from top to botton  # decide to save the full image
        self.resolution = [768, 576]

        # TODO: turn this one to visualize
        # Visualization related
        self.show_screen = False  # NON CARLA SETTINGS PARAM
        self.aspect_ratio = [3, 1]  # only matters when showing the screen
        self.scale_factor = 1  # NON CARLA SETTINGS PARAM
        self.plot_vbp = False
        self.number_screens = 1 # added later

        # others
        self.interface = "Carla" # always fixed
