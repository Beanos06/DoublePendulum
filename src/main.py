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
double_pend_parameters = DPendulumParameters(
    l1 = L_1,
    l2 = L_2,
    m1 = M_1,
    m2 = M_2
)
# Parameters for sliding pendulum
sliding_pend_parameters = SSPendulumParameters(
    l = 1.5,
    m1 = 5,
    m2 = 5,
)

initial_conditions = [THETA1, VEL1, THETA2, VEL2]
initial_conditions2 = [0, 0, 0, 0]
pendulums = []

double_pendulum = DoublePendulum(
    initial_conditions=initial_conditions,
    func=double_pendulum_ODE,
    params=double_pend_parameters,
    dt=0.01,
    color=(255,255,255)
)
pendulums.append(double_pendulum)

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
change_sim_button = Button((195, 20), "Change Simulation",  [close_button.size[0] + 20, 0+V_PADDING], screen, bgColor=(173, 216, 230), txtColor=(0,0,0))

pendulum_id = 0

# Main game loop
while True:
    events = pygame.event.get()
    pendulum = pendulums[pendulum_id]
    
    screen.fill((0,0,0))
    reset_button.render()
    close_button.render()
    change_sim_button.render()
    
    pendulum.update()
    pendulum.draw(screen, CENTER)
    pendulum.display_data(screen, CENTER, 12)

    if reset_button.clicked(events):
        print(f"Reset {pendulum}")
        pendulum.reset()
    
    if change_sim_button.clicked(events):
        pendulum_id += 1
        if pendulum_id >= len(pendulums):
            pendulum_id = 0
    
    if close_button.clicked(events):
        pygame.quit()
        print("Closed application")
        exit()

    keys = pygame.key.get_pressed()

    # Moving the cart left or right
    if keys[pygame.K_LEFT]:
        pendulum.force = -50
    elif keys[pygame.K_RIGHT]:
        pendulum.force = 50
    else:
        pendulum.force = 0
    
    pygame.display.update()
    clock.tick(100)