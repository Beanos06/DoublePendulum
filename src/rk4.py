def rk4_step(f, y, t, dt, *args):
    r1 = f(t, y, *args)
    r2 = f(t + dt/2, y + dt/2 * r1, *args)
    r3 = f(t + dt/2, y + dt/2 * r2, *args)
    r4 = f(t + dt, y + dt * r3, *args)

    return y + (r1 + 2*r2 + 2*r3 + r4) * dt / 6