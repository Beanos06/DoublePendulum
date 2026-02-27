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
        x1 = center[0] - self.params.l1 * 100 * math.sin(self.theta1)
        y1 = center[1] + self.params.l2 * 100 * math.cos(self.theta1)

        # Coordinates of second mass (x2, y2)
        x2 = x1 - self.params.l2 * 100 * math.sin(self.theta2)
        y2 = y1 + self.params.l2 * 100 * math.cos(self.theta2)

        pygame.draw.circle(screen, self.color, (x0,y0), radius=6)
        pygame.draw.line(screen, self.color, (x0,y0), (x1,y1), 2)
        pygame.draw.circle(screen, self.color, (x1, y1), radius=6)
        pygame.draw.line(screen, self.color, (x1, y1), (x2, y2), 2)
        pygame.draw.circle(screen, self.color, (x2, y2), radius=6)

@dataclass
class SSPendulumParameters:
    m1: float
    m2: float
    l: float
    g: float = 9.81

class SlidingSimplePendulum:
    def __init__(
            self,
            initial_conditions: list,
            func: Callable,
            params: SSPendulumParameters,
            dt: float,
            color,
        ):
        self.y = np.array(initial_conditions, dtype=float)
        self.x, self.vel_x, self.theta, self.vel_theta = self.y
        self.t = 0
        self.force = 0

        self.initial_conditions = initial_conditions
        self.func = func
        self.params = params
        self.dt = dt
        self.color = color

    def update(self):
        """
        Update state of double pendulum for timestep dt
        """
        
        self.y = rk4_step(
            lambda t, y, params: self.func(t, y, params, self.force),
            self.y,
            self.t,
            self.dt,
            self.params
        )
        self.t += self.dt
        self.x, self.vel_x, self.theta, self.vel_theta = self.y
        return self.y
    
    def draw(self, screen, center):
        """
        Draws current state of pendulum on pygame screen
        """

        # Coordinates of the cart
        x0 = center[0] + self.x * 100
        y0 = center[1]

        # Coordinates of suspended mass
        x1 = x0 - self.params.l * 100 * math.sin(self.theta)
        y1 = y0 + self.params.l * 100 * math.cos(self.theta)

        pygame.draw.line(screen, (150,150,150), (0, center[1]), (center[0] * 2, center[1]), 1)
        pygame.draw.circle(screen, self.color, (x0,y0), radius=12)
        pygame.draw.line(screen, self.color, (x0,y0), (x1,y1), 2)
        pygame.draw.circle(screen, self.color, (x1, y1), radius=6)