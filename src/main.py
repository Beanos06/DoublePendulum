from pendulum import DoublePendulum, DPendulumParameters
from physics.double_pendulum import double_pendulum_ODE
import numpy as np

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

DP1 = DoublePendulum(
    initial_conditions=initial_conditions,
    params=parameters,
    func=double_pendulum_ODE,
    dt=0.01,
)

for i in range(0, 100):
    y = DP1.update()

    print(y)