from pendulum import DoublePendulum, DPendulumParameters
from physics.double_pendulum import double_pendulum_ODE
import numpy as np
import math
import pygame
from sys import exit

# Pygame Initialization

pygame.init()
WIDTH = 1000
HEIGHT = 600
CENTER = (WIDTH/2, HEIGHT/2)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Double Pendulum Simulation")
clock = pygame.time.Clock()

# Initial Conditions and Parameters of the Double Pendulum
G = 9.81
THETA1 = np.pi
THETA2 = 2
VEL1 = 0
VEL2 = 0
L_1 = 1
L_2 = 1
M_1 = 5
M_2 = 5

parameters = DPendulumParameters(
    l1 = L_1,
    l2 = L_2,
    m1 = M_1,
    m2 = M_2
)

initial_conditions = [THETA1, VEL1, THETA2, VEL2]

dbl_pend = DoublePendulum(
    initial_conditions=initial_conditions,
    params=parameters,
    func=double_pendulum_ODE,
    dt=0.01,
)


y = dbl_pend.load()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
    
    screen.fill((0,0,0))
    
    y = dbl_pend.update()
    
    pygame.draw.circle(
        screen,
        (225,225,225),
        CENTER,
        radius=6
    )
    pygame.draw.line(
        screen, 
        (255,255,255), 
        CENTER, 
        (
            CENTER[0] + dbl_pend.params.l1 * 100 * math.cos(y[0] + np.pi/2),
            CENTER[1] + dbl_pend.params.l1 * 100 * math.sin(y[0] + np.pi/2)
        ), 
        2
    )
    pygame.draw.circle(
        screen,
        (225,225,0),
        (
            CENTER[0] + dbl_pend.params.l1 * 100 * math.cos(y[0] + np.pi/2),
            CENTER[1] + dbl_pend.params.l1 * 100 * math.sin(y[0] + np.pi/2)         
        ),
        radius=6
    )
    
    pygame.draw.line(
        screen, 
        (255,255,255), 
        (
            CENTER[0] + dbl_pend.params.l1 * 100 * math.cos(y[0] + np.pi/2),
            CENTER[1] + dbl_pend.params.l1 * 100 * math.sin(y[0] + np.pi/2)
        ), 
        (
            CENTER[0] + dbl_pend.params.l1 * 100 * math.cos(y[0] + np.pi/2) + dbl_pend.params.l2 * 100 * math.cos(y[2] + np.pi/2),
            CENTER[1] + dbl_pend.params.l1 * 100 * math.sin(y[0] + np.pi/2) + dbl_pend.params.l2 * 100 * math.sin(y[2] + np.pi/2)
        ), 
        2
    )
    
    pygame.draw.circle(
        screen,
        (225,225,0),
        (
            CENTER[0] + dbl_pend.params.l1 * 100 * math.cos(y[0] + np.pi/2) + dbl_pend.params.l2 * 100 * math.cos(y[2] + np.pi/2),
            CENTER[1] + dbl_pend.params.l1 * 100 * math.sin(y[0] + np.pi/2) + dbl_pend.params.l2 * 100 * math.sin(y[2] + np.pi/2)         
        ),
        radius=6
    )
    
    pygame.display.update()
    clock.tick(60)