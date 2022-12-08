IDM_DELTA = 4.0

class IDM:

    @staticmethod
    def compute_acceleration(a_max, 
                                a_pref, 
                                v_curr, 
                                v_target, 
                                pos_delta, 
                                vel_delta, 
                                min_space, 
                                time_pref,
                                delta_time):

        '''
        Compute acceleration of a vehicle based on IDM model.
        
        1. a_max: Maximum acceleration
        2. a_pref: Preferred acceleration (or deceleration)
        3. v_curr: Current velocity of ego vehicle
        4. v_target: Target velocity of ego vehicle
        5. pos_delta: Position delta from leading vehicle
        6. vel_delta: Velocity delta from leading vehicle
        7. min_space: Minimum desired distance to leading vehicle
        8. time_pref: Desired time to move forward with current speed
        9. delta_time: Delta time used for preventing negative velocity
        '''
        
        optimal_spacing = (min_space + v_curr * time_pref + \
            ((v_curr * vel_delta) / (2 * pow(a_max * a_pref, 0.5))))

        # @BUGFIX: Concentually, optimal spacing cannot be
        # negative value. If it is allowed to be, acceleration
        # could become negative value even when the leading
        # vehicle is much faster than ego vehicle, so it can
        # accelerate more. It results in unrealistic behavior
        # and degenerates solution of control problem.

        clipped_optimal_spacing = (optimal_spacing < 0.0)
        optimal_spacing = max(optimal_spacing, 0)

        acc = a_max * (1.0 - pow(v_curr / v_target, IDM_DELTA) \
            - pow((optimal_spacing / pos_delta), 2.0))

        # prevent negative velocity;

        clipped_acceleration = (acc < -v_curr / delta_time)
        acc = max(acc, -v_curr / delta_time)

        return acc, optimal_spacing, clipped_acceleration, clipped_optimal_spacing