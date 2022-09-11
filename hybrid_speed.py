import time
import torch as th
from math import floor

from road.road_network import RoadNetwork
from road.lane.macro_lane import MacroLane
from road.lane.micro_lane import MicroLane

# params
micro_portion = 0.2

seg_length = 10000.0
num_seg = 15

macro_cell_length = 1000.0
speed_limit = 30.0

delta_time = 0.03
num_step = 100

# simulation
network = RoadNetwork(speed_limit)

for i in range(num_seg):
    macro_lane_length = seg_length * (1.0 - micro_portion)
    micro_lane_length = seg_length - macro_lane_length

    # macro
    lane = MacroLane(macro_lane_length, speed_limit, macro_cell_length)
    for j in range(lane.num_cell):
        lane.curr_rho[[j], :] = 0.7
        lane.curr_u[[j], :] = speed_limit * 0.7
    network.add_lane(lane)

    # micro
    lane = MicroLane(micro_lane_length, speed_limit)
    num_vehicle = floor(lane.length / 15.0)
    for j in range(num_vehicle):
        pos, vel, a_max, a_pref, v_target, min_space, time_pref, vehicle_length = lane.rand_vehicle()
        pos[0, 0] = lane.length - 1e-3 - j * 15
        vel[0, 0] = 0.0
        vehicle_a = th.ones((1, 1)) * vehicle_length
        lane.add_vehicle(pos, vel, a_max, a_pref, v_target, min_space, time_pref, vehicle_length, vehicle_a)
    network.add_lane(lane)

# run
start_time = time.time()

for step in range(num_step):
    network.forward_step(delta_time, False)

    print("Step {}".format(step))
    network.print()

end_time = time.time()

print("Elapsed time: {:2f} sec".format(end_time - start_time))
print("Frame Per Second: {:2f}".format(num_step / (end_time - start_time)))