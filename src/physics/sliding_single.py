from sympy import diff, sin, cos
import sympy as sm

t = sm.symbols('t')
m1, m2, l, g = sm.symbols('m_1,m_2,l,g')

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

V = -m2 * g * cos(theta)

L = T - V

EL_1 = diff(L, x) - diff(diff(L, x_d), t).simplify()
EL_2 = diff(L, theta) - diff(diff(L, theta_d), t).simplify()

solutions = sm.solve([EL_1, EL_2], x_dd, theta_dd)
