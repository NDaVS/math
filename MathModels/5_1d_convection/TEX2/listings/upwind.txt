def upwind(uu, c, dx, dt):
    u_ = uu.copy()
    k = c * dt / dx
    for ti in range(u_.shape[0] - 1):
        for xi in range(1, u_.shape[1] - 1):
            u_[ti + 1, xi] = u_[ti, xi] + k * (u_[ti, xi - 1] - u_[ti, xi])
    return u_
