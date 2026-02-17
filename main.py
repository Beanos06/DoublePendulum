from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
from pendulum import DoublePendulum, DPendulumParameters
import numpy as np

# Create time points and initial conditions
time_points = np.linspace(0, 20, 1000)
G = 9.81
THETA1 = np.pi
THETA2 = 2
VEL1 = 0
VEL2 = 0
L_1 = 1
L_2 = 1
M_1 = 5
M_2 = 5

fig, ax = plt.subplots()
plt.axis('equal')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

parameters = DPendulumParameters(
    l1 = L_1,
    l2 = L_2,
    m1 = M_1,
    m2 = M_2
)
DP1 = DoublePendulum(
    initial_conditions=[THETA1, VEL1, THETA2, VEL2],
    params=parameters,
    time_points=time_points,
    ax=ax,
    color='blue'
)

theta1 = DP1.load()

# Matplotlib animation
def update(frame):
    DP1.update(frame)
    

animation = FuncAnimation(fig, update, frames=len(time_points), interval=15)

# Run the following code to save the simulation as a gif
# animation.save("simulation.gif", writer=PillowWriter(fps=25))

plt.show()
    