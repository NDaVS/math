def TDMA(a, b, c, f):
    a, b, c, f = tuple(map(lambda k_list: list(map(float, k_list)), (a, c, b, f)))
    alpha = [-b[0] / c[0]]
    beta = [f[0] / c[0]]
    n = len(f)
    x = [0] * n
    for i in range(1, n):
        denom = a[i] * alpha[i - 1] + c[i]
        alpha.append(-b[i] / denom)
        beta.append((f[i] - a[i] * beta[i - 1]) / denom)
    x[n - 1] = beta[n - 1]
    for i in range(n - 1, 0, -1):
        x[i - 1] = alpha[i - 1] * x[i] + beta[i - 1]
    return x


def implicit(uu, v, dx, dt):
    m = uu.shape[0] - 1
    n = uu.shape[1] - 1
    u_ = uu.copy()
    k = v * dt / (2 * dx)
    for ti in range(m):
        al = np.zeros(n + 1)
        bl = np.ones(n + 1)
        cl = np.zeros(n + 1)
        al[1:-1] = -k
        cl[1:-1] = k
        bl[0] = bl[-1] = 1
        u_[ti + 1] = TDMA(al, bl, cl, u_[ti])
    return u_
