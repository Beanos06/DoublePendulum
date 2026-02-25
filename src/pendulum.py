"""
Pendulum Classes
"""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from rk4 import rk4_step
from typing import Callable

# Dataclass to store the parameters for the double pendulum
@dataclass
class DPendulumParameters:
    l1: float
    l2: float
    m1: float
    m2: float
    g: float = 9.81
    
class DoublePendulum:
    def __init__(
            self, 
            initial_conditions: list, 
            func: Callable[[float, NDArray, DPendulumParameters], NDArray], 
            params: DPendulumParameters, 
            dt: float
        ):
        self.y = np.array(initial_conditions, dtype=float)
        self.theta1, self.vel1, self.theta2, self.vel2 = self.y
        
        self.params = params
        self.func = func
        self.t = 0
        self.dt = dt
    
    def load(self):
        return self.y

    def update(self):
        self.y = rk4_step(
            self.func,
            self.y,
            self.t,
            self.dt,
            self.params
        )
        self.t += self.dt
        self.theta1, self.vel1, self.theta2, self.vel2 = self.y
        return self.y