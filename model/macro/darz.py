from asyncio import InvalidStateError
import torch as th
import numpy as np
from model.macro._arz import ARZ, GAMMA, EPSILON

class dARZ(ARZ):
    
    '''
    Extension of [ARZ] for computing analytical gradients.
    '''

    @staticmethod
    def compute_dL():
        '''
        Compute partial derivative of the left state values 
        w.r.t the left and right cell state values.
        
        dL / dL =
            [ d(rho_L) / d(rho_L), d(rho_L) / d(y_L)  ]
            [ d(y_L) / d(rho_L),   d(y_L) / d(y_L)    ]

        dL / dR = 
            [ 0, 0 ]
            [ 0, 0 ]
        '''

        # dL / dL
        dL = np.eye(2, dtype=np.float32)

        # dL / dR
        dR = np.zeros((2, 2), dtype=np.float32)

        return dL, dR

    @staticmethod
    def compute_dM(Q_m: ARZ.FullQ, Q_L: ARZ.FullQ, Q_R: ARZ.FullQ, u_max, gamma=GAMMA):
        '''
        Compute partial derivative of Q Middle state values 
        w.r.t the left and right cell state values.
        
        dM / dL =
            [ d(rho_M) / d(rho_L), d(rho_M) / d(y_L)  ]
            [ d(y_M) / d(rho_L),   d(y_M) / d(y_L)    ]

        dM / dR = 
            [ d(rho_M) / d(rho_L), d(rho_M) / d(y_L)  ]
            [ d(y_M) / d(rho_L),   d(y_M) / d(y_L)    ]
        '''

        r_L = max(Q_L.q.r, EPSILON)       # Prevent divide by zero
        r_R = max(Q_R.q.r, EPSILON)       
        y_L = Q_L.q.y
        y_R = Q_R.q.y
        u_L = Q_L.u
        u_R = Q_R.u

        r_M = Q_m.q.r
        u_M = Q_m.u
        ueq_M = Q_m.u_eq
        ueq_prime_M = ARZ.compute_u_eq_prime(r_M, u_max, gamma)

        # preliminaries;

        # d(u_L) / d(r_L)
        duL_drL = -y_L / (r_L ** 2) + ARZ.compute_u_eq_prime(r_L, u_max, gamma)
        # d(u_L) / d(y_L)
        duL_dyL = 1.0 / r_L
        # d(u_R) / d(r_R)
        duR_drR = -y_R / (r_R ** 2) + ARZ.compute_u_eq_prime(r_R, u_max, gamma)
        # d(u_R) / d(y_R)
        duR_dyR = 1.0 / r_R

        # ===================

        # dM / dL
        # d(r_M) / d(r_L)
        a = (1.0 / gamma) * (r_M ** (1.0 - gamma))
        b = gamma * (r_L ** (gamma - 1.0))
        c = (1.0 / u_max) * (duL_drL)
        drM_drL = a * (b + c)

        # d(r_M) / d(y_L)
        d = (1.0 / u_max) * (duL_dyL)
        drM_dyL = a * d

        # d(y_M) / d(r_L)
        e = u_M - ueq_M
        dyM_drL = drM_drL * e + r_M * (-ueq_prime_M * drM_drL)
        
        # d(y_M) / d(y_L)
        dyM_dyL = drM_dyL * e + r_M * (-ueq_prime_M * drM_dyL)
        
        # ===================

        # dM / dR
        # d(r_M) / d(r_R)
        f = (-1.0 / u_max) * duR_drR
        drM_drR = a * f

        # d(r_M) / d(y_R)
        g = (-1.0 / u_max) * duR_dyR
        drM_dyR = a * g

        # d(y_M) / d(r_R)
        dyM_drR = drM_drR * e + r_M * (duR_drR - ueq_prime_M * drM_drR)
        
        # d(y_M) / d(y_R)
        dyM_dyR = drM_dyR * e + r_M * (duR_dyR - ueq_prime_M * drM_dyR)

        dL = np.zeros((2, 2), dtype=np.float32)
        dL[0][0] = drM_drL
        dL[0][1] = drM_dyL
        dL[1][0] = dyM_drL
        dL[1][1] = dyM_dyL

        dR = np.zeros((2, 2), dtype=np.float32)
        dR[0][0] = drM_drR
        dR[0][1] = drM_dyR
        dR[1][0] = dyM_drR
        dR[1][1] = dyM_dyR
        
        return dL, dR

    @staticmethod
    def compute_dC(Q_c: ARZ.FullQ, Q_L: ARZ.FullQ, u_max, gamma=GAMMA):
        '''
        Compute partial derivative of Centered Rarefaction 
        w.r.t the left and right cell state values.
        
        dC_dL =
            [ d(rho_0) / d(rho_L), d(rho_0) / d(y_L)  ]
            [ d(y_0) / d(rho_L),   d(y_0) / d(y_L)    ]
        
        dC_dR =
            [ 0, 0 ]
            [ 0, 0 ]
        '''

        r_L = max(Q_L.q.r, EPSILON)       # Prevent divide by zero
        y_L = Q_L.q.y
        ueq_prime_L = ARZ.compute_u_eq_prime(r_L, u_max, gamma)

        r_C = Q_c.q.r
        u_C = Q_c.u
        ueq_C = Q_c.u_eq
        ueq_prime_C = ARZ.compute_u_eq_prime(r_C, u_max, gamma)

        # preliminaries;
        # d(u_L) / d(rho_L)
        duL_drL = -y_L / (r_L ** 2) + ueq_prime_L
        
        # d(u_L) / d(y_L)
        duL_dyL = 1.0 / (r_L)

        f = u_max * gamma * (r_L ** (gamma - 1.0))

        # d(u_c) / d(r_L)
        duC_drL = (gamma / (gamma + 1)) * (duL_drL + f)

        # d(u_c) / d(y_L)
        duC_dyL = (gamma / (gamma + 1)) * duL_dyL

        # ===================

        # d(r_C) / d(r_L)
        #a = Q_L.u + u_max * (Q_L.q.rho ** gamma)
        b = (gamma + 1) * u_max
        #c = (a / b) ** ((1.0 - gamma) / gamma)
        # (Q_c.q.rho) equals to (a / b).
        c = r_C ** (1.0 - gamma)
        d = c / gamma
        e = d / b
        drC_drL = e * (duL_drL + f)

        # d(rho_0) / d(y_L)
        drC_dyL = e * (duL_dyL)

        # d(y_0) / d(rho_L)
        g = u_C - ueq_C
        dyC_drL = drC_drL * g + r_C * (duC_drL - ueq_prime_C * drC_drL)

        # d(y_0) / d(y_L)
        dyC_dyL = drC_dyL * g + r_C * (duC_dyL - ueq_prime_C * drC_dyL)

        dL = np.zeros((2, 2), dtype=np.float32)
        dL[0][0] = drC_drL
        dL[0][1] = drC_dyL
        dL[1][0] = dyC_drL
        dL[1][1] = dyC_dyL

        dR = np.zeros((2, 2), dtype=np.float32)
        return dL, dR

    @staticmethod
    def compute_dLdR(rs: ARZ.Riemann, Q_L: ARZ.FullQ, Q_R: ARZ.FullQ, u_max, gamma=GAMMA):
        '''
        Compute partial derivatives (dQ0 / dQL) and (dQ0 / dQR) using [rs].
        Use [rs.case_ind] to redirect gradient computation to right case.
        '''

        if rs.case_ind == 0:

            return dARZ.compute_dL()

        elif rs.case_ind == 1:

            return dARZ.compute_dM(rs.Q_0, Q_L, Q_R, u_max, gamma)

        elif rs.case_ind == 2:

            return dARZ.compute_dC(rs.Q_0, Q_L, u_max, gamma)

        else:

            raise ValueError("Case index of Riemann solution should be 0 (Q_L), 1 (Q_M) or 2 (Q_C)")

    @staticmethod
    def flux_prime(q: ARZ.FullQ):
        '''
        Compute jacobian matrix of flux w.r.t. the given state.
        '''
        fp = np.zeros((2, 2), dtype=np.float32)

        r = max(q.q.r, EPSILON)         # Prevent divide by zero
        y = q.q.y
        ueq = q.u_eq
        ueq_prime = dARZ.compute_u_eq_prime(r, q.u_max)
        
        fp[0][0] = ueq + r * ueq_prime
        fp[0][1] = 1
        fp[1][0] = y * ueq_prime - ((y / r) ** 2)
        fp[1][1] = (2.0 * y) / r + ueq

        return fp