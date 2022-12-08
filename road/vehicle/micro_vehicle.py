from road.vehicle.vehicle import Vehicle

class MicroVehicle(Vehicle):
    
    '''
    Vehicle that is used in microscopic traffic simulation.

     1. position: Position of the vehicle in a lane
     2. speed: Speed of the vehicle (not vector)
     3. accel_max: Maximum acceleration (per second)
     4. accel_pref: Preferred acceleration (per second)
     5. target_speed: Target speed, often set as nearly speed limit of the road
     6. min_space: Minimum desired distance to leading vehicle
     7. time_pref: Desired time to move forward with current speed
     8. length: Vehicle length
    '''

    def __init__(self, position, speed, accel_max, accel_pref, target_speed, min_space, time_pref, length, a):

        super().__init__(position, speed, length, a)

        self.accel_max = accel_max
        self.accel_pref = accel_pref
        self.target_speed = target_speed
        self.min_space = min_space
        self.time_pref = time_pref