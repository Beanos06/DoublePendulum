"""
Pendulum Classes
"""

from dataclasses import dataclass
import numpy as np
from scipy.integrate import odeint
from physics import system_of_odes

@dataclass
class DPendulumParameters:
    l1: float
    l2: float
    m1: float
    m2: float
    g: float = 9.81
    
class DoublePendulum:
    def __init__(self, initial_conditions, params, time_points, ax, color):
        self.initial_conditions = initial_conditions
        self.theta1 = initial_conditions[0]
        self.vel1 = initial_conditions[1]
        self.theta2 = initial_conditions[2]
        self.vel2 = initial_conditions[3]
        
        self.params = params
        self.time_points = time_points
        self.ax = ax
        self.color = color
        
        self.solution = odeint(
            system_of_odes,
            self.initial_conditions,
            time_points,
            args=(
                params.l1, 
                params.l2,
                params.m1,
                params.m2, 
                params.g
            )
        )
        
        self.the1_sol = self.solution[:, 0]
        self.the1_d_sol = self.solution[:, 1]

        self.the2_sol = self.solution[:, 2]
        self.the2_d_sol = self.solution[:, 3]

        self.x1 = params.l1 * np.sin(self.the1_sol)
        self.y1 = -params.l1 * np.cos(self.the1_sol)

        self.x2 = self.x1 + params.l1 * np.sin(self.the2_sol)
        self.y2 = self.y1 - params.l1 * np.cos(self.the2_sol)
        
    
    def load(self, display_data=False):    
        # Load in the double pendulum
        self.pendulum1, = self.ax.plot([0, self.x1[0]], [0, self.y1[0]], color=self.color, lw=2)
        self.mass1, = self.ax.plot([self.x1[0]], [self.y1[0]], 'o', markersize=2*self.params.m1, color=self.color)

        self.pendulum2, = self.ax.plot([self.x1[0], self.x2[0]], [self.y1[0], self.y2[0]], color=self.color, lw=2)
        self.mass2, = self.ax.plot([self.x2[0]], [self.y2[0]], 'o', markersize=2*self.params.m2, color=self.color)
        
        self.display_data = display_data
        if display_data:
            self.theta1_text = self.ax.text(
                0.95, 0.05, f'θ1: {self.the1_sol[0]}',
                verticalalignment='bottom', horizontalalignment='right',
                transform=self.ax.transAxes,
                color='green', fontsize=10)
            self.theta2_text = self.ax.text(
                0.95, 0.01, f'θ2: {self.the2_sol[0]}',
                verticalalignment='bottom', horizontalalignment='right',
                transform=self.ax.transAxes,
                color='green', fontsize=10)

        
    def update(self, frame):
        # Update the positions of the masses and pendulum legs for each frame
        self.pendulum1.set_data([0, self.x1[frame]], [0, self.y1[frame]])
        self.mass1.set_data([self.x1[frame]], [self.y1[frame]])

        self.pendulum2.set_data([self.x1[frame], self.x2[frame]], [self.y1[frame], self.y2[frame]])
        self.mass2.set_data([self.x2[frame]], [self.y2[frame]])
        
        if self.display_data:
            self.theta1_text.set_text(f'θ1: {self.the1_sol[frame]:.3f}')
            self.theta2_text.set_text(f'θ2: {self.the2_sol[frame]:.3f}')
        
        return  self.pendulum1, self.mass1, self.pendulum2, self.mass2    