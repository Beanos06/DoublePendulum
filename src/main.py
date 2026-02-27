from pendulum import DoublePendulum, DPendulumParameters, SSPendulumParameters, SlidingSimplePendulum
from physics.double_pendulum import double_pendulum_ODE
from physics.sliding_single import sliding_simple_pendulum_ODE
import numpy as np
import pygame
from sys import exit
from components.button import Button

# Pygame Initialization

pygame.init()
WIDTH = 1000
HEIGHT = 600
CENTER = (WIDTH/2, HEIGHT/2)
H_PADDING = 10
V_PADDING = 10
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pendulum Simulation")
clock = pygame.time.Clock()

# Initial Conditions and Parameters of the Double Pendulum
G = 9.81
THETA1 = np.pi/2
THETA2 = 0
VEL1 = 0
VEL2 = 0
L_1 = 1
L_2 = 1
M_1 = 5
M_2 = 5

# Parameters for double pendulum
parameters = DPendulumParameters(
    l1 = L_1,
    l2 = L_2,
    m1 = M_1,
    m2 = M_2
)

sliding_pend_parameters = SSPendulumParameters(
    l = 1,
    m1 = 5,
    m2 = 5,
)

initial_conditions = [THETA1, VEL1, THETA2, VEL2]
initial_conditions2 = [0, 0, 0, 0]
pendulums = []

sliding_pendulum = SlidingSimplePendulum(
    initial_conditions2,
    sliding_simple_pendulum_ODE,
    sliding_pend_parameters,
    dt=0.01,
    color=(255,255,255),
)
pendulums.append(sliding_pendulum)

reset_button = Button((60,20), "Reset", [0+H_PADDING,0+V_PADDING], screen, bgColor=(255,237,41), txtColor=(0,0,0))
close_button = Button((60,20), "Close", [WIDTH-60-H_PADDING, 0+V_PADDING], screen, bgColor=(255,44,44), txtColor=(0,0,0))

# Main game loop
while True:
    events = pygame.event.get()
    
    screen.fill((0,0,0))
    
    for dbl_pend in pendulums:
        dbl_pend.update()
        dbl_pend.draw(screen, CENTER)

        if reset_button.clicked(events):
            print("Reset")
            dbl_pend.reset()
        
        if close_button.clicked(events):
            pygame.quit()
            exit()
            print("Closed application")

    reset_button.render()
    close_button.render()

    keys = pygame.key.get_pressed()

    # Moving the cart left or right
    for pend in pendulums:
        if keys[pygame.K_LEFT]:
            pend.force = -30
        elif keys[pygame.K_RIGHT]:
            pend.force = 30
        else:
            pend.force = 0
    
    pygame.display.update()
    clock.tick(100)