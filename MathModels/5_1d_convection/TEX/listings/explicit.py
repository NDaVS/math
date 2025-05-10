def explicit(uu, v, dx, dt):
    m = uu.shape[0] - 1
    n = uu.shape[1] - 1
    u_ = uu.copy()
    k = v * dt / (2 * dx)

    for ti in range(m):
        for j in range(1, n):
            u_[ti + 1, j] = u_[ti, j] - k * (u_[ti, j + 1] - u_[ti, j - 1])

        u_[ti + 1, 0] = u_[ti, 0]
        u_[ti + 1, n] = u_[ti, n]

    return u_
