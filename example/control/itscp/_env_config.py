default_config = {

    # number of intersections;
        
    "num_intersection": 3,

    # number of lanes for each road;

    "num_lane": 3,

    # lane length (m);

    "lane_length": 40,

    # speed limit (m/s);

    "speed_limit": 30,

    # cell length (m);

    "cell_length": 5,

    # vehicle length (m);

    "vehicle_length": 5,

    # camera center poisition;

    "centering_position": [0.5, 0.5],

    # screen width and height;

    "screen_width": 1440,
    "screen_height": 1440,

    # screen scale;

    "screen_scale": -3,

    # simulation frequency: number of frames that are drawn per second;
    
    "simulation_frequency": 30,

    # policy length: time length (sec) that one action persist;
    
    "policy_length": 30,

    # signal length: time length (sec) that one signal sequence persist;
    
    "signal_length": 10,

    # action bounds;

    "action_min": 0.3,
    "action_max": 0.7,

    # number of actions for one episode;

    "duration": 1,

    # threshold for determining static cell: used for reward calculation;

    "static_speed": 0.2,

    # number of schedule information to observe;

    "num_schedule_obs": 10,

    # maximum number of micro vehicles generated per lane in [micro] mode;

    "max_num_micro_vehicle_per_lane": 10,

    # simulation mode, can be [macro], [micro], or [hybrid];

    "mode": "hybrid",   # "macro"

    # render;

    "render": False,

    # random seed, use it if it is larger than 0;

    "random_seed": 0,

}