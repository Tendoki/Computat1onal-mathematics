import numpy as np
import random as rand
def LUPQ(A, n):
    L = np.zeros((n, n))
    U = np.copy(A)
    P = np.eye(n)
    Q = np.eye(n)
    rank = n
    for i in range(n):
        max = U[i][i]
        for j in range(i, n):
            for k in range(i, n):
                if abs(U[j][k]) >= abs(max):
                    max = U[j][k]
                    index = j, k
        if (abs(U[index[0], index[1]]) < 1e-14):
            rank = i
        if index[0] != i:
            for j in range(n):
                L[index[0]][j], L[i][j] = L[i][j], L[index[0]][j]
                U[index[0]][j], U[i][j] = U[i][j], U[index[0]][j]
                P[index[0]][j], P[i][j] = P[i][j], P[index[0]][j]
        if index[1] != i:
            for j in range(n):
                L[j][index[1]], L[j][i] = L[j][i], L[j][index[1]]
                U[j][index[1]], U[j][i] = U[j][i], U[j][index[1]]
                Q[j][index[1]], Q[j][i] = Q[j][i], Q[j][index[1]]

        for j in range(i + 1, n):
            L[j][i] = U[j][i] / U[i][i]
            U[j] -= U[i] * L[j][i]
    for j in range(n):
        L[j][j] = 1
    return L, U, P, Q, rank

def Determinant(U, n):
    det = 1
    for i in range(n):
        det *= U[i][i]
    return det

def Solution(L, U, P, Q, b, n):
    #Ly = Pb, Uz = y, x = Qz
    y = np.matmul(P, b)
    for i in range(1, n):
        y[i] -= sum([L[i][k] * y[k] for k in range(0, i)])
    z = y
    for i in range(n - 1, -1, -1):
        z[i] -= sum([U[i][k] * z[k] for k in range(n-1, i, -1)])
        z[i] /= U[i][i]
    return np.matmul(Q, z)

def inverse(L, U, P, Q, n):
    A_inverse = np.eye(n)
    E = np.eye(n)
    for i in range(n):
        A_inverse[i] = Solution(L, U, P, Q, E[i], n)
    A_inverse = [[A_inverse[j][i] for j in range(n)] for i in range(n)]
    return A_inverse

def norma(A):
    norma = 0
    for i in range(A.shape[0]):
        n = 0
        for j in range(A.shape[1]): n += abs(A[i, j])
        if (n > norma): norma = n
    return norma

def norma1(A, A1):
    t = norma(A) * norma(A1)
    return t

n = 5
A = np.random.random((n, n))
b = np.array(np.random.sample(n))
# A = np.array([[1.,1.,1.],[2.,2.,2.],[3.,3.,3.]], dtype=float)
# n = 5
# A = np.array([[1., 2., -1., 2.,-5.],
#               [2., -3., 2, 4.,13.],
#               [1., 2., -1., 2.,-5.],
#               [-1., -6., 7, -3., 7.],
#               [4., 9., -43., 1., 12.]], dtype=float)
# b = [4.,1.,9.,2.,13.]
# b = np.array(np.random.sample(n))
# n = 4
# A = np.array([[1., 2., -1., 2.],
#               [2., -3., 2, 4.],
#               [3., 1., 1., 6.],
#               [-1., -6., 7, -3]], dtype=float)
print('A =')
print(A,'\n')
L, U, P, Q, rank = LUPQ(A, n)
print('LU =')
print(np.matmul(L, U),'\n')
print('L =')
print(L,'\n')
print('U =')
print(U,'\n')
print('rank =', rank,'\n')
print('Перестановка строк P =')
print(P,'\n')
print('Перестановка столбцов Q =')
print(Q,'\n')
print('PAQ =')
print(np.matmul(np.matmul(P,A), Q),'\n')
print('LU - PAQ =')
LUPAQ = np.matrix(np.matmul(L, U)) - np.matrix(np.matmul(P, np.matmul(A, Q)))
if np.all(abs(LUPAQ)) < 10 ** (-14):
    LUPAQ = np.zeros((n, n))
print(LUPAQ,'\n')
print('Определитель:')
print(Determinant(U, n),'\n')
x = Solution(L, U, P, Q, b, n)
print('Решение СЛАУ:')
print(x,'\n')
print('Python решение СЛАУ:')
print(np.linalg.solve(A, b),'\n')
print("Ax - b =", np.subtract(np.matmul(A, x), b), '\n')
A1 = np.array(inverse(L, U, P, Q, n))
print('Обратная матрица')
print(A1,'\n')
print('Python обратная матрица')
print(np.linalg.inv(A), '\n')
print("A*invA=\n", np.matmul(A, A1), '\n')
print("invA*A=\n", np.matmul(A1, A), '\n')
print('Число обусловленности')
print(norma1(A, A1))