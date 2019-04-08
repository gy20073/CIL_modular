class configDrive:
    def __init__(self):
        # human or machine
        self.type_of_driver = "Human"

        # resource related
        self.host = "127.0.0.1"
        self.port = 2000
        self.path = "/scratch/yang/aws_data/carla_collect/rfs_sim_v6/"  # If path is set go for it , if not expect a name set

        # data collection related
        self.city_name = 'RFS_MAP'
        self.carla_config = None # This will be filled by the caller
        # collect method
        self.autopilot = True
        self.use_planner = True # only useful in carlaHuman, used to randomly walk in the city
        self.noise = "Spike" #"Spike"  # NON CARLA SETTINGS PARAM
        # TODO: some spike related numbers

        self.reset_period = 960 # reset when the system time goes beyond this number
        # Those parameters will override carla_config
        self.weather = "1" # This will be override by the caller
        self.cars = "50"
        self.pedestrians = "100"
        # TODO: change hash_data_collection
        self.num_images_to_collect = 200*20 * 3 # how many images to collect in total
        self.re_entry = True # True to allow continue collecting the data, this will make changes to the folder structure

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

        '''
        # noiser related params
        self.noise_frequency = 45
        self.noise_intensity = 2.5 # 0.5 - 1.5 scaled by this factor
        self.min_noise_time_amount = 1.0
        self.no_noise_decay_stage = True
        self.use_tick = True
        self.time_amount_multiplier = 0.0 # original 0.5 second to 2 second, this converts to 0.15 second to 0.6 second
        self.noise_std = 0.0
        self.no_time_offset = True

        # the actual noise is: 0.03 * 0.5 * self.intensity
        # self.noise_time_amount = self.min_noise_time_amount
        '''

        '''
        # noiser related params
        self.noise_frequency = 45 # FIXED, something has enough noise
        self.noise_intensity = 15.0 * 2
        self.min_noise_time_amount = 0.5
        self.no_noise_decay_stage = True #FIXED
        self.use_tick = True # FIXED
        # originally default values
        self.time_amount_multiplier = 1.0 # the smaller the better
        self.noise_std=0.0 # same effect as self.intensity, IGNORE
        self.no_time_offset = True
        # 0.03 * this_intensity = 0.03 * 7.5 =  0.225
        '''
        self.noise_frequency = 45
        self.noise_intensity = 5
        self.min_noise_time_amount = 0.5
        self.no_noise_decay_stage = True
        self.use_tick = True

        self.positions_file = "town03_positions/merged.csv"
        if self.city_name == "RFS_MAP":
            self.parking_position_file = "town03_intersections/positions_file_RFS_MAP.parking.txt"
            # there are some bug in this function, that the whole simulation will get stuck
            # disable the over-exploration since we solve the shoulder by marking it on the map
            self.extra_explore_prob = 0.0 # now for debugging purpose, let all of them be the extra positions file
            #self.extra_explore_position_file = "town03_intersections/positions_file_RFS_MAP.extra_explore.txt"
            self.extra_explore_position_file = "town03_intersections/positions_file_RFS_MAP.extra_explore_v3.txt"
            self.extra_explore_location_std = 2.0
            self.extra_explore_yaw_std = 20.0
