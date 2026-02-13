import sympy as sm
from sympy import diff, sin, cos

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
y_1 = -l_1 * sin(theta1)

# Second mass' positions
x2 = l_1 * sin(theta1) + l_2 * sin(theta2)
y_2 = -l_1 * sin(theta1) - l_2 * sin(theta2)

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
x2_d = theta2_d * l_2 * cos(theta2) + y1_d

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