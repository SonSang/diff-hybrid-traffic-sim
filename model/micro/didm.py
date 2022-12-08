import torch as th
import math

from model.micro._idm import IDM, IDM_DELTA

class dIDM(IDM):

    '''
    Extension of IDM for computing analytical gradients.
    '''

    @staticmethod
    def compute_dEgo(   a_max, 
                        a_pref, 
                        v_curr, 
                        v_target, 
                        pos_delta, 
                        vel_delta, 
                        min_space, 
                        time_pref,
                        optimal_spacing,
                        delta_time,
                        clipped_acceleration,
                        clipped_optimal_spacing):
        
        '''
        Compute partial derivative of the ego vehicle 
        state w.r.t the the ego vehicle state of last
        time step.

        @ clipped_acceleration: True when acceleration has
        been clipped to prevent neg. speed in forward pass.

        @ clipped_optimal_spacing: True when optimal spacing
        has been clipped to zero in forward pass.
        '''

        dEgo = th.zeros((2, 2))
        
        dEgo[0][0] = 1
        dEgo[0][1] = delta_time

        if not clipped_acceleration:

            dEgo[1][0] = delta_time * (-2 * a_max * (math.pow(optimal_spacing, 2) / math.pow(pos_delta, 3)))

            if clipped_optimal_spacing:
                dEgo[1][1] = 1 + delta_time * a_max * \
                    (-IDM_DELTA * (math.pow(v_curr, (IDM_DELTA - 1)) / math.pow(v_target, IDM_DELTA)))
            else:
                dEgo[1][1] = 1 + delta_time * a_max * \
                    (-IDM_DELTA * (math.pow(v_curr, (IDM_DELTA - 1)) / math.pow(v_target, IDM_DELTA)) \
                        - 2 * (optimal_spacing / math.pow(pos_delta, 2)) * \
                            (time_pref + ((v_curr + vel_delta) / (2 * math.sqrt(a_max * a_pref)))))

        return dEgo

    @staticmethod
    def compute_dLeading(   a_max, 
                            a_pref, 
                            v_curr, 
                            v_target, 
                            pos_delta, 
                            vel_delta, 
                            min_space, 
                            time_pref,
                            optimal_spacing,
                            delta_time,
                            clipped_acceleration,
                            clipped_optimal_spacing):

        '''
        Compute partial derivative of the ego vehicle 
        state w.r.t the the leading vehicle state of last
        time step.

        @ clipped_acceleration: True when acceleration has
        been clipped to prevent neg. speed in forward pass.
        
        @ clipped_optimal_spacing: True when optimal spacing
        has been clipped to zero in forward pass.
        '''

        dLeading = th.zeros((2, 2))
        
        dLeading[0][0] = 0
        dLeading[0][1] = 0

        if not clipped_acceleration:

            dLeading[1][0] = delta_time * (2 * a_max * (math.pow(optimal_spacing, 2) / math.pow(pos_delta, 3)))

            if clipped_optimal_spacing:

                dLeading[1][1] = delta_time * a_max * \
                    (- 2 * (optimal_spacing / math.pow(pos_delta, 2)))

            else:
                dLeading[1][1] = delta_time * a_max * \
                    (- 2 * (optimal_spacing / math.pow(pos_delta, 2)) * \
                            (-v_curr / (2 * math.sqrt(a_max * a_pref))))

        return dLeading