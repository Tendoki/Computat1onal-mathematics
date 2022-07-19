import numpy as np
import random as r

def QR(A):
    R = np.copy(A)
    Q = np.eye(R.shape[0])
    T = np.eye(R.shape[0])
    for i in range(A.shape[0]):
        for j in range(i + 1, A.shape[0]):
            Temp = np.eye(A.shape[0])
            s = R[j, i] / np.sqrt(R[i, i] ** 2 + R[j, i] ** 2)
            c = R[i, i] / np.sqrt(R[i, i] ** 2 + R[j, i] ** 2)
            Temp[i, i] = c
            Temp[i, j] = s
            Temp[j, i] = -s
            Temp[j, j] = c
            R = np.matmul(Temp, R)
            T = np.matmul(Temp, T)
            Temp[i, j], Temp[j, i] = Temp[j, i], Temp[i, j]
            Q = np.matmul(Q, Temp)
    return Q, R, T

size = r.randint(2, 5)
A = np.matrix([[r.randint(0, 100) for j in range(size)] for i in range(size)])
Q, R, T = QR(A)
print("A\n", A)
print("Q=\n", Q)
print("R=\n", R)
print("QR=\n", np.matmul(Q, R))
b = np.transpose(np.matrix([r.randint(0, 100) for j in range(R.shape[0])]))
print('b=',b)
y = np.matmul(T, b)
x = np.copy(y)
for i in range(size-1, -1, -1):
    x[i] -= sum([R[i][k] * x[k] for k in range(size-1, i, -1)])
    x[i] /= R[i][i]
print('x=',x)
print('x=',np.linalg.solve(A,b))