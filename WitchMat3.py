import numpy as np
import numpy
import math
import time

count = 0

def F(x):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = x.tolist()
    return np.transpose(numpy.mat([
        math.cos(x2 * x1) - math.exp(-3 * x3) + x4 * x5 ** 2 - x6 - math.sinh(
            2 * x8) * x9 + 2 * x10 + 2.0004339741653854440,
        math.sin(x2 * x1) + x3 * x9 * x7 - math.exp(-x10 + x6) + 3 * x5 ** 2 - x6 * (x8 + 1) + 10.886272036407019994,
        x1 - x2 + x3 - x4 + x5 - x6 + x7 - x8 + x9 - x10 - 3.1361904761904761904,
        2 * math.cos(-x9 + x4) + x5 / (x3 + x1) - math.sin(x2 ** 2) + math.cos(
            x7 * x10) ** 2 - x8 - 0.1707472705022304757,
        math.sin(x5) + 2 * x8 * (x3 + x1) - math.exp(-x7 * (-x10 + x6)) + 2 * math.cos(x2) - 1.0 / (
                    -x9 + x4) - 0.3685896273101277862,
        math.exp(x1 - x4 - x9) + x5 ** 2 / x8 + math.cos(3 * x10 * x2) / 2 - x6 * x3 + 2.0491086016771875115,
        x2 ** 3 * x7 - math.sin(x10 / x5 + x8) + (x1 - x6) * math.cos(x4) + x3 - 0.7380430076202798014,
        x5 * (x1 - 2 * x6) ** 2 - 2 * math.sin(-x9 + x3) + 0.15e1 * x4 - math.exp(
            x2 * x7 + x10) + 3.5668321989693809040,
        7 / x6 + math.exp(x5 + x4) - 2 * x2 * x8 * x10 * x7 + 3 * x9 - 3 * x1 - 8.4394734508383257499,
        x10 * x1 + x9 * x2 - x8 * x3 + math.sin(x4 + x5 + x6) * x7 - 0.78238095238095238096]
    ))


def J(x):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = x.tolist()
    return numpy.mat([[-x2 * math.sin(x2 * x1), -x1 * math.sin(x2 * x1), 3 * math.exp(-3 * x3), x5 ** 2, 2 * x4 * x5,
                       -1, 0, -2 * math.cosh(2 * x8) * x9, -math.sinh(2 * x8), 2],
                      [x2 * math.cos(x2 * x1), x1 * math.cos(x2 * x1), x9 * x7, 0, 6 * x5,
                       -math.exp(-x10 + x6) - x8 - 1, x3 * x9, -x6, x3 * x7, math.exp(-x10 + x6)],
                      [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
                      [-x5 / (x3 + x1) ** 2, -2 * x2 * math.cos(x2 ** 2), -x5 / (x3 + x1) ** 2, -2 * math.sin(-x9 + x4),
                       1.0 / (x3 + x1), 0, -2 * math.cos(x7 * x10) * x10 * math.sin(x7 * x10), -1,
                       2 * math.sin(-x9 + x4), -2 * math.cos(x7 * x10) * x7 * math.sin(x7 * x10)],
                      [2 * x8, -2 * math.sin(x2), 2 * x8, 1.0 / (-x9 + x4) ** 2, math.cos(x5),
                       x7 * math.exp(-x7 * (-x10 + x6)), -(x10 - x6) * math.exp(-x7 * (-x10 + x6)), 2 * x3 + 2 * x1,
                       -1.0 / (-x9 + x4) ** 2, -x7 * math.exp(-x7 * (-x10 + x6))],
                      [math.exp(x1 - x4 - x9), -1.5 * x10 * math.sin(3 * x10 * x2), -x6, -math.exp(x1 - x4 - x9),
                       2 * x5 / x8, -x3, 0, -x5 ** 2 / x8 ** 2, -math.exp(x1 - x4 - x9),
                       -1.5 * x2 * math.sin(3 * x10 * x2)],
                      [math.cos(x4), 3 * x2 ** 2 * x7, 1, -(x1 - x6) * math.sin(x4),
                       x10 / x5 ** 2 * math.cos(x10 / x5 + x8),
                       -math.cos(x4), x2 ** 3, -math.cos(x10 / x5 + x8), 0, -1.0 / x5 * math.cos(x10 / x5 + x8)],
                      [2 * x5 * (x1 - 2 * x6), -x7 * math.exp(x2 * x7 + x10), -2 * math.cos(-x9 + x3), 1.5,
                       (x1 - 2 * x6) ** 2, -4 * x5 * (x1 - 2 * x6), -x2 * math.exp(x2 * x7 + x10), 0,
                       2 * math.cos(-x9 + x3),
                       -math.exp(x2 * x7 + x10)],
                      [-3, -2 * x8 * x10 * x7, 0, math.exp(x5 + x4), math.exp(x5 + x4),
                       -7.0 / x6 ** 2, -2 * x2 * x8 * x10, -2 * x2 * x10 * x7, 3, -2 * x2 * x8 * x7],
                      [x10, x9, -x8, math.cos(x4 + x5 + x6) * x7, math.cos(x4 + x5 + x6) * x7,
                       math.cos(x4 + x5 + x6) * x7, math.sin(x4 + x5 + x6), -x3, x2, x1]])

def LUPQ(A):
    global count
    L = np.zeros((len(A), len(A)))
    U = np.copy(A)
    P = np.eye(len(A))
    Q = np.eye(len(A))
    for i in range(len(A)):
        max = U[i][i]
        for j in range(i, len(A)):
            for k in range(i, len(A)):
                if abs(U[j][k]) >= abs(max):
                    max = U[j][k]
                    index = j, k
        if index[0] != i:
            for j in range(len(A)):
                L[index[0]][j], L[i][j] = L[i][j], L[index[0]][j]
                U[index[0]][j], U[i][j] = U[i][j], U[index[0]][j]
                P[index[0]][j], P[i][j] = P[i][j], P[index[0]][j]
        if index[1] != i:
            for j in range(len(A)):
                L[j][index[1]], L[j][i] = L[j][i], L[j][index[1]]
                U[j][index[1]], U[j][i] = U[j][i], U[j][index[1]]
                Q[j][index[1]], Q[j][i] = Q[j][i], Q[j][index[1]]

        for j in range(i + 1, len(A)):
            L[j][i] = U[j][i] / U[i][i]
            U[j] -= U[i] * L[j][i]
            count += len(U[j]) + 1
    for j in range(len(A)):
        L[j][j] = 1
    return L, U, P, Q

def Solution(L, U, P, Q, b):
    global count
    #Ly = Pb, Uz = y, x = Qz
    y = np.matmul(P, b)
    for i in range(len(y)):
        y[i] -= sum([L[i][k] * y[k] for k in range(0, i)])
        count += i
    z = y
    for i in range(len(z) - 1, -1, -1):
        z[i] -= sum([U[i][k] * z[k] for k in range(len(y)-1, i, -1)])
        z[i] /= U[i][i]
        count += len(y) - 1 + 1
    return np.matmul(Q, np.squeeze(np.asarray(z)))

def Newton(eps, k, hybrid, full):
    # x = np.array([0.5,0.5,1.5,-1.0,-0.5,1.5,0.5,-0.5,1.5,-1.5])
    x = np.array([0.5, 0.5, 1.5, -1.0, -0.2, 1.5, 0.5, -0.5, 1.5, -1.5])
    i = 0
    while (True and i < 1000):
        if (not full and hybrid and i % k == 0 or not full and not hybrid and i <= k or full):
            Jacobi = J(x)
            L, U, P, Q = LUPQ(Jacobi)
        else:
            print("modification")
        print("x" + str(i) + " =", x)
        p = Solution(L, U, P, Q, -F(x))
        xk = np.add(x, p)
        delta = np.linalg.norm(np.subtract(x, xk))
        x = np.asarray(xk)
        i += 1

        if (delta < eps): return x

start = time.time()
x = Newton(1e-14, 5, True, False)
print("\nВремя решения:", time.time() - start)
print("\nКол-во операций", count, "\n")
print("F(x) = ", F(x))