from sympy import diff, sin, cos
import sympy as sm
from numpy.typing import NDArray
from pendulum import SSPendulumParameters
import numpy as np

t = sm.symbols('t')
m1, m2, l, g = sm.symbols('m_1,m_2,l,g')

# x is the position of the sliding cart
x=sm.symbols('x', cls=sm.Function)
theta = sm.symbols(r'\theta', cls=sm.Function)
x = x(t)
theta = theta(t)

x2 = x + l*sin(theta)
y2 = -l*cos(theta)

x_d = diff(x,t)
x_dd = diff(x_d, t)
theta_d = diff(theta,t)
theta_dd = diff(theta_d,t)

x2_d = x_d + theta_d * l * cos(theta)
y2_d = theta_d * l * sin(theta)

# Lagrangian
T_1 = 0.5 * m1 * x_d**2
T_2 = 0.5 * m2 * (x2_d**2 + y2_d**2)

T = T_1 + T_2

V = -m2 * g * l * cos(theta)

L = T - V

F = sm.symbols('F')

EL_1 = diff(L, x) - diff(diff(L, x_d), t) - F
EL_2 = diff(L, theta) - diff(diff(L, theta_d), t).simplify()

solutions = sm.solve([EL_1, EL_2], x_dd, theta_dd)

EL_fn1 = sm.lambdify((x, x_d, theta, theta_d, t, l, m1, m2, g, F),solutions[x_dd])
EL_fn2 = sm.lambdify((x, x_d, theta, theta_d, t, l, m1, m2, g, F),solutions[theta_dd])

def sliding_simple_pendulum_ODE(
        t: float,
        y: NDArray, 
        params: SSPendulumParameters,
        F: float = 0
    ) -> NDArray:
    """
    System of ODEs representing the motion of a double pendulum
    """
    
    x, x_d, theta, theta_d = y

    l = params.l
    m1 = params.m1
    m2 = params.m2
    g = params.g
    
    # x_dd = EL_fn1(x, x_d, theta, theta_d, t, l, m1, m2, g, F)
    # theta_dd = EL_fn2(x, x_d, theta, theta_d, t, l, m1, m2, g, F)

    x_dd = F / m1
    theta_dd = -(params.g * np.sin(theta) - x_dd * np.cos(theta)) / params.l
    
    # 3. Add some "air resistance" to the pendulum so it eventually stops swinging
    damping = 0.2
    theta_dd -= damping * theta_d

    return np.array([x_d, x_dd, theta_d, theta_dd])