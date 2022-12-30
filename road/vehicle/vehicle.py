DEFAULT_VEHICLE_LENGTH = 5.0

class Vehicle:
    '''
    Basic vehicle that is used in traffic simulation.

     1. position: Position of the vehicle in a lane
     2. speed: Speed of the vehicle (not vector)
     3. length: Length of the vehicle
     4. a: Ancillary variable for gradient flow
    '''
    def __init__(self, id, position, speed, length, a):

        self.id = id
        self.position = position
        self.speed = speed
        self.length = length
        self.a = a