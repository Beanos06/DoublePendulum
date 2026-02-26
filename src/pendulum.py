"""
Pendulum Classes
"""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from rk4 import rk4_step
from typing import Callable
import pygame
import math

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
            dt: float,
            color
        ):
        self.y = np.array(initial_conditions, dtype=float)
        self.theta1, self.vel1, self.theta2, self.vel2 = self.y
        
        self.params = params
        self.func = func
        self.t = 0
        self.dt = dt
        self.color = color

    def update(self):
        """
        Update state of double pendulum for timestep dt
        """
        
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
    
    def draw(self, screen, center):
        """
        Draws current state of double pendulum on pygame screen
        """

        # Coordinates of anchor point (x0, y0)
        x0, y0 = center

        # Coordinates of first mass (x1, y1)
        x1 = center[0] - self.params.l1 * 100 * math.cos(self.theta1 + np.pi/2)
        y1 = center[1] + self.params.l2 * 100 * math.sin(self.theta1 + np.pi/2)

        # Coordinates of second mass (x2, y2)
        x2 = x1 - self.params.l2 * 100 * math.cos(self.theta2 + np.pi/2)
        y2 = y1 + self.params.l2 * 100 * math.sin(self.theta2 + np.pi/2)

        pygame.draw.circle(screen, self.color, (x0,y0), radius=6)
        pygame.draw.line(screen, self.color, (x0,y0), (x1,y1), 2)
        pygame.draw.circle(screen, self.color, (x1, y1), radius=6)
        pygame.draw.line(screen, self.color, (x1, y1), (x2, y2), 2)
        pygame.draw.circle(screen, self.color, (x2, y2), radius=6)