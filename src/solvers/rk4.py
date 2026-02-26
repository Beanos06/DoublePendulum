def rk4_step(f, x, t, dt):
    k1 = f(x, t)
    k2 = f(x + k1*dt/2, t + dt/2)
    k3 = f(x + k2*dt/2, t + dt/2)
    k4 = f(x + k3*dt, t + dt)
    
    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)