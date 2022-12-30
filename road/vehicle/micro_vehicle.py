import numpy as np

from road.vehicle.vehicle import Vehicle, DEFAULT_VEHICLE_LENGTH

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

    def __init__(self, id, position, speed, accel_max, accel_pref, target_speed, min_space, time_pref, length, a):

        super().__init__(id, position, speed, length, a)

        self.accel_max = accel_max
        self.accel_pref = accel_pref
        self.target_speed = target_speed
        self.min_space = min_space
        self.time_pref = time_pref

    @staticmethod
    def default_micro_vehicle(speed_limit: float):

        '''
        Generate a default micro vehicle, of which attributes depend on the speed limit.
        '''

        # vehicle length;

        vehicle_length = DEFAULT_VEHICLE_LENGTH

        # maximum acceleration;

        a_max = speed_limit * 1.75

        # preferred acceleration;
        
        a_pref = speed_limit * 1.25

        # target speed;
        
        target_speed = speed_limit * 1.0

        # min space ahead;
        
        min_space = vehicle_length * 0.3

        # preferred time to go;
        
        time_pref = 0.4

        vehicle = MicroVehicle(-1, 
                                0, 
                                0, 
                                a_max, 
                                a_pref, 
                                target_speed, 
                                min_space, 
                                time_pref, 
                                vehicle_length, 
                                vehicle_length)

        return vehicle

    @staticmethod
    def random_micro_vehicle(speed_limit: float):

        '''
        Generate a micro vehicle with randomly chosen attributes, which depend on the speed limit.
        '''

        # vehicle length;
        # @TODO: apply dynamic sizes;

        vehicle_length = DEFAULT_VEHICLE_LENGTH

        # maximum acceleration;

        a_max = np.random.rand()
        a_max = np.interp(a_max, [0, 1], [speed_limit * 1.5, speed_limit * 2.0])
        
        # preferred acceleration;
        
        a_pref = np.random.rand()
        a_pref = np.interp(a_pref, [0, 1], [speed_limit * 1.0, speed_limit * 1.5])
        
        # target speed;
        
        target_speed = np.random.rand()
        target_speed = np.interp(target_speed, [0, 1], [speed_limit * 0.8, speed_limit * 1.2])
        
        # min space ahead;
        
        min_space = np.random.rand()
        min_space = np.interp(min_space, [0, 1], [vehicle_length * 0.2, vehicle_length * 0.4])
        
        # preferred time to go;
        
        time_pref = np.random.rand()
        time_pref = np.interp(time_pref, [0, 1], [0.2, 0.6])

        vehicle = MicroVehicle(-1, 
                                0, 
                                0, 
                                a_max, 
                                a_pref, 
                                target_speed, 
                                min_space, 
                                time_pref, 
                                vehicle_length, 
                                vehicle_length)

        return vehicle