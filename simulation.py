from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sympy import diff, sin, cos
import sympy as sm
import numpy as np

# Create the sympy variables
t = sm.symbols('t')
m_1, m_2, l_1, l_2, g = sm.symbols('m_1,m_2,l_1,l_2,g', positive=True)

# Set two functions theta1 and theta2
theta1, theta2 = sm.symbols(r'\theta_1, \theta_2', cls=sm.Function)

# Specifying the variable
theta1 = theta1(t)
theta2 = theta2(t)

# First mass' positions
x1 = l_1 * sin(theta1)
y_1 = -l_1 * cos(theta1)

# Second mass' positions
x2 = l_1 * sin(theta1) + l_2 * sin(theta2)
y_2 = -l_1 * cos(theta1) - l_2 * cos(theta2)

#### Important derivatives
# Angular speed 
theta1_d = diff(theta1, t)
theta2_d = diff(theta2, t)
# Angular acceleration 
theta1_dd = diff(theta1_d, t)
theta2_dd = diff(theta2_d, t)

#### Basic kinematics
x1_d = theta1_d * l_1 * sin(theta1)
y1_d = theta1_d * l_1 * cos(theta1)
x2_d = theta2_d * l_2 * sin(theta2) + x1_d
y2_d = theta2_d * l_2 * cos(theta2) + y1_d

# Kinetic Energy (T)
T1 = 0.5 * m_1 * theta1_d**2 * l_1**2
T2 = 0.5 * m_2 * (theta1_d**2 * l_1**2 + 2 * theta1_d * theta2_d * l_1 * l_2 * cos(theta1 - theta2) + theta2_d**2 * l_2**2)
T = T1 + T2

# Potention Energy (V)
V1 = -m_1 * g * l_1 * cos(theta1)
V2 = -m_2 * g * (l_1 * cos(theta1) + l_2 * cos(theta2))
V = V1 + V2

# The Lagrangian (L)
L = T - V

dL_dtheta1 = diff(L, theta1)
dL_dtheta1_d = diff(L, theta1_d)
EL_eq1 = ( dL_dtheta1 - diff(dL_dtheta1_d, t) ).simplify()

dL_dtheta2 = diff(L, theta2)
dL_dtheta2_d = diff(L, theta2_d)
EL_eq2 = ( dL_dtheta2 - diff(dL_dtheta2_d, t) ).simplify()

# Solve the two Euler-Lagrange equations for theta1_dd, theta2_dd
# since we need first order differential equations
solutions = sm.solve([EL_eq1, EL_eq2], theta1_dd, theta2_dd)

# Turn theta1_dd and theta2_dd into useable functions
EL_fn1 = sm.lambdify((theta1, theta2, theta1_d, theta2_d, t, l_1, l_2, m_1, m_2, g), solutions[theta1_dd])
EL_fn2 = sm.lambdify((theta1, theta2, theta1_d, theta2_d, t, l_1, l_2, m_1, m_2, g), solutions[theta2_dd])

# Create time points and initial conditions
time_points = np.linspace(0, 20, 1000)
G = 9.81
THETA1 = 3
THETA2 = 2
VEL1 = 0
VEL2 = 0
INITIAL_CONDITION = [THETA1, VEL1, THETA2, VEL2] # Angle 1, velocity 1, angle 2, velocity 2
L_1 = 1
L_2 = 1
M_1 = 5
M_2 = 5

def system_of_odes(y, t, l_1, l_2, m_1, m_2, g):
    theta1, theta1_d, theta2, theta2_d = y
    
    theta1_dd = EL_fn1(theta1, theta2, theta1_d, theta2_d, t, l_1, l_2, m_1, m_2, g)
    theta2_dd = EL_fn2(theta1, theta2, theta1_d, theta2_d, t, l_1, l_2, m_1, m_2, g)
    
    return [theta1_d, theta1_dd, theta2_d, theta2_dd]

solution = odeint(system_of_odes, INITIAL_CONDITION, time_points, args=(L_1, L_2, M_1, M_2, G))

the1_sol = solution[:, 0]
the1_d_sol = solution[:, 1]

the2_sol = solution[:, 2]
the2_d_sol = solution[:, 3]

x1_pendulum = L_1 * np.sin(the1_sol)
y1_pendulum = -L_1 * np.cos(the1_sol)

x2_pendulum = x1_pendulum + L_2 * np.sin(the2_sol)
y2_pendulum = y1_pendulum - L_2 * np.cos(the2_sol)

# Matplotlib animation
def update(frame):
    pendulum1.set_data([0, x1_pendulum[frame]], [0, y1_pendulum[frame]])
    mass1.set_data([x1_pendulum[frame]], [y1_pendulum[frame]])

    pendulum2.set_data([x1_pendulum[frame], x2_pendulum[frame]], [y1_pendulum[frame], y2_pendulum[frame]])
    mass2.set_data([x2_pendulum[frame]], [y2_pendulum[frame]])

    return pendulum1, mass1, pendulum2, mass2

fig, ax = plt.subplots()
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

pendulum1, = ax.plot([0, x1_pendulum[0]], [0, y1_pendulum[0]], color='blue', lw=2)
mass1, = ax.plot([x1_pendulum[0]], [y1_pendulum[0]], 'o', markersize=2*M_1, color='blue')

pendulum2, = ax.plot([x1_pendulum[0], x2_pendulum[0]], [y1_pendulum[0], y2_pendulum[0]], color='blue', lw=2)
mass2, = ax.plot([x2_pendulum[0]], [y2_pendulum[0]], 'o', markersize=2*M_2, color='blue')

animation = FuncAnimation(fig, update, frames=len(time_points), interval=20, blit=True)

plt.show()
    