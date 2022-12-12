GAMMA = 0.5
EPSILON = 1e-3

class ARZ:
    
    '''
    Classes and functions used in ARZ model.
    '''
    
    class Q:
        
        '''
        State of a single cell in the lane that contains density (r) and relative flow(y).
        
        Density (r) : Density of traffic, i.e. "cars per car length".
        Relative flow (y) : Relative flow of traffic.
        '''
        
        def __init__(self):
            self.r = 0
            self.y = 0

        @staticmethod
        def from_r_y(r, y):
            q = ARZ.Q()
            q.r = r
            q.y = y
            return q

        @staticmethod
        def from_r_u(r, u, u_max):
            q = ARZ.Q()
            q.r = r
            q.y = ARZ.compute_y(r, u, u_max)
            return q

        def __mul__(self, c: float):
            return ARZ.Q.from_r_y(self.r * c, self.y * c)

        def __add__(self, q):
            return ARZ.Q.from_r_y(self.r + q.r, self.y + q.y)

        def __sub__(self, q):
            return ARZ.Q.from_r_y(self.r - q.r, self.y - q.y)

        def clear(self):
            self.r = 0
            self.y = 0
       
    class FullQ: 
        
        '''
        Extension of Q, including speed (u) and speed limit (u_max) information.
        
        Speed: Speed of traffic flow.
        '''
        
        def __init__(self, u_max):
            self.q = ARZ.Q()
            self.u_max = u_max
            self.u = u_max
            self.u_eq = u_max

        @staticmethod
        def from_q(q, u_max):
            fq = ARZ.FullQ(u_max)
            fq.q = q
            fq.u_max = u_max
            fq.u = ARZ.compute_u(q.r, q.y, u_max)
            fq.u_eq = ARZ.compute_u_eq(q.r, u_max)
            return fq

        @staticmethod
        def from_r_u(r, u, u_max):
            fq = ARZ.FullQ(u_max)
            fq.q = ARZ.Q.from_r_u(r, u, u_max)
            fq.u_max = u_max
            fq.u = u
            fq.u_eq = ARZ.compute_u_eq(r, u_max)
            return fq

        def set_r_u(self, r, u, u_max):
            self.u_max = u_max
            self.u = u
            self.q = ARZ.Q.from_r_u(r, u, u_max)
            self.u_eq = ARZ.compute_u_eq(r, u_max)

        def set_r_y(self, r, y, u_max):
            self.u_max = u_max
            self.u = ARZ.compute_u(r, y, u_max)
            self.q = ARZ.Q.from_r_y(r, y)
            self.u_eq = ARZ.compute_u_eq(r, u_max)

        def flux(self):
            return ARZ.Q.from_r_y(self.flux_r(), self.flux_y())

        def flux_r(self):
            return self.q.r * self.u

        def flux_y(self):
            return self.q.y * self.u

        def lambda_0(self):
            return self.u + self.q.r * (ARZ.compute_u_eq_prime(self.q.r, self.u_max))

        def lambda_1(self):
            return self.u

        def __sub__(self, fq):
            return ARZ.FullQ.from_q(self.q - fq.q, self.u_max)

        def clear(self):
            self.q.clear()
            self.u = self.u_max
            self.u_eq = self.u_max

    '''
    Basic helper methods that can be used in solving ARZ system.
    '''

    @staticmethod
    def compute_y(r, u, u_max):
        u_eq = ARZ.compute_u_eq(r, u_max)
        return r * (u - u_eq)

    @staticmethod
    def compute_u(r, y, u_max):
        r = max(r, EPSILON)              # prevent divide by zero
        u_eq = ARZ.compute_u_eq(r, u_max)
        u = (y / r) + u_eq
        return u

    @staticmethod
    def compute_u_eq(r, u_max, gamma=GAMMA):
        r = max(r, EPSILON)              # prevent divide by zero in backward
        return u_max * (1.0 - pow(r, gamma))

    @staticmethod
    def compute_r_from_u_eq(u_eq, u_max, gamma=GAMMA):
        u_max = max(u_max, EPSILON)
        gamma = max(gamma, EPSILON)          # prevent divide by zero
        return pow(1.0 - u_eq / u_max, 1.0 / gamma)

    @staticmethod
    def compute_u_eq_prime(r, u_max, gamma=GAMMA):
        r = max(r, EPSILON)
        return -u_max * gamma * pow(r, gamma - 1)
    
    '''
    Helper methods to solve Riemann problem.
    '''

    @staticmethod
    def compute_Ql(Q_L: FullQ):
        
        '''
        Q Left.
        '''

        nQ = ARZ.FullQ(Q_L.u_max)
        nQ.set_r_y(Q_L.q.r, Q_L.q.y, Q_L.u_max)     # same as Q_L

        return nQ

    @staticmethod
    def compute_Qc(Q_L: FullQ, u_max, gamma=GAMMA):
    
        '''
        Centered Rarefaction.
        '''
    
        Q_c = ARZ.FullQ(u_max)
        Q_c.u_max = u_max

        Q_c.q.r = ((Q_L.u + u_max * (Q_L.q.r ** gamma)) / ((gamma + 1) * u_max)) ** (1.0 / gamma)
        Q_c.u = (gamma / (gamma + 1)) * (Q_L.u + u_max * (Q_L.q.r ** gamma))
        Q_c.q.y = ARZ.compute_y(Q_c.q.r, Q_c.u, u_max)
        Q_c.u_eq = ARZ.compute_u_eq(Q_c.q.r, u_max, gamma)

        return Q_c

    @staticmethod
    def compute_Qm(Q_L: FullQ, Q_R: FullQ, u_max, gamma=GAMMA):

        '''
        Q Middle.
        '''

        Q_m = ARZ.FullQ(u_max)
        Q_m.u_max = u_max

        Q_m.q.r = ((Q_L.q.r ** gamma) + ((Q_L.u - Q_R.u) / u_max)) ** (1.0 / gamma)
        Q_m.u = Q_R.u
        Q_m.q.y = ARZ.compute_y(Q_m.q.r, Q_m.u, u_max)
        Q_m.u_eq = ARZ.compute_u_eq(Q_m.q.r, u_max, gamma)

        return Q_m

    class Riemann:
        '''
        Data structure to store solution of Riemann problem.
        '''
        def __init__(self):
            
            self.speed0 = 0
            self.speed1 = 0
            self.Q_0 = ARZ.FullQ(0)
            self.case_ind = -1      # 0 = Q_L, 1 = Q_M, 2 = Q_C

    @staticmethod
    def riemann_solve(Q_L: FullQ, Q_R: FullQ, u_max):
        
        '''
        Given two Q states from left (x < 0) and right (x > 0),
        compute the state at the middle (x = 0) in advanced time (t > 0).
        This intermediate state would be used to compute flux between
        given two cells, and update their states in macro simulation.
        '''

        solution: ARZ.Riemann = ARZ.Riemann()
        
        # Case 4 : r_l == 0 (Left Q is almost vacuum)
        if Q_L.q.r < EPSILON:

            solution.speed0 = 0.0
            solution.speed1 = Q_L.u

            solution.case_ind = 0  
        
        # Case 5 : r_r == 0 (Right Q is almost vacuum)
        elif Q_R.q.r < EPSILON:
            
            Q_m = ARZ.FullQ.from_r_u(0.0, u_max + Q_L.u - Q_L.u_eq, u_max)
            
            lambda_0_l = Q_L.lambda_0()
            lambda_0_m = Q_m.u

            solution.speed0 = (lambda_0_l + lambda_0_m) * 0.5
            solution.speed1 = solution.speed0

            if lambda_0_l >= 0.0:

                solution.case_ind = 0

            else:

                solution.case_ind = 2

        # Case 0 : u_l == u_r
        elif abs(Q_L.u - Q_R.u) < EPSILON:
            
            solution.speed0 = 0.0
            solution.speed1 = Q_R.u

            solution.case_ind = 0

        # Case 1 : u_l > u_r
        elif Q_L.u > Q_R.u:
        
            Q_m = ARZ.compute_Qm(Q_L, Q_R, u_max)
            flux_r_diff = Q_m.flux_r() - Q_L.flux_r()

            solution.speed0 = flux_r_diff / max(Q_m.q.r - Q_L.q.r, EPSILON)
            solution.speed1 = Q_R.u
            
            if solution.speed0 >= 0.0:
            
                solution.case_ind = 0
            
            else:
            
                solution.case_ind = 1
        
        # Case 2 : u_max + u_l - Q_L.u_eq > u_r
        elif u_max + Q_L.u - Q_L.u_eq > Q_R.u:
        
            Q_m = ARZ.compute_Qm(Q_L, Q_R, u_max)
            lambda_0_l = Q_L.lambda_0()
            lambda_0_m = Q_m.lambda_0()

            solution.speed0 = (lambda_0_l + lambda_0_m) * 0.5
            solution.speed1 = Q_R.u
        
            if lambda_0_l >= 0:
            
                solution.case_ind = 0
            
            elif lambda_0_m <= 0:
            
                solution.case_ind = 1
            
            else:
            
                solution.case_ind = 2
        
        # Case 3 : u_l <= u_r - u_max * (r_l ** gamma)
        else:

            Q_m = ARZ.FullQ.from_r_u(0.0, u_max + Q_L.u - Q_L.u_eq, u_max)
            lambda_0_l = Q_L.lambda_0()
            lambda_0_m = Q_m.u

            solution.speed0 = (lambda_0_l + lambda_0_m) * 0.5
            solution.speed1 = Q_R.u
        
            if lambda_0_l >= 0.0:
            
                solution.case_ind = 0
            
            else:
            
                solution.case_ind = 2

        if solution.case_ind == 0:

            solution.Q_0 = ARZ.compute_Ql(Q_L)

        elif solution.case_ind == 1:
            
            solution.Q_0 = ARZ.compute_Qm(Q_L, Q_R, u_max)

        elif solution.case_ind == 2:

            solution.Q_0 = ARZ.compute_Qc(Q_L, u_max)

        else:

            assert False, "Invalid case index in Riemann solution"

        return solution