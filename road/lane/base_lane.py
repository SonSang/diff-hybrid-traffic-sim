'''
@ author: SonSang (Sanghyun Son)
@ email: shh1295@gmail.com
'''
class BaseLane:
    '''
    Lane, 0 for macro lane, 1 for micro lane
    '''
    def __init__(self, length: float, speed_limit: float):
        self.next_lane: BaseLane = None
        self.prev_lane: BaseLane = None
        
        self.length = length
        self.speed_limit = speed_limit

    def is_macro(self):
        raise NotImplementedError()

    def is_micro(self):
        raise NotImplementedError()