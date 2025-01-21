import numpy as np

for _ in range(100):
    n = 8
    A = np.random.rand(n, n) * 10

    first_norm = np.linalg.norm(A, ord=1)
    second_norm = np.linalg.norm(A, ord='fro')
    infinity_norm = np.linalg.norm(A, ord=np.inf)
    o_sum = 0
    for i in range(n):
        t_sum = 0
        for j in range(n):
            t_sum += np.abs(A[i, j])
        o_sum += t_sum ** 2

    # print("Matrix A:")
    # print(A)Я
    # print(f"||A||_1: {first_norm}")
    # print(f"||A||_2: {second_norm}")
    # print(f"||A||_e: {infinity_norm}")
    # print(f"Equivalence ratio: {second_norm ** 2} <= {first_norm * infinity_norm} ")
    print(f'Is equivalence true: {True if second_norm <= infinity_norm * first_norm else False}')