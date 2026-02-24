def rk4_step(f, y, t, dt):
    r1 = f(t, y)
    r2 = f(t + dt/2, y + dt/2 * r1)
    r3 = f(t + dt/2, y + dt/2 * r2)
    r4 = f(t + dt, y + dt * r3)

    next_step = y + (r1 + 2*r2 + 2*r3 + r4) * dt / 6