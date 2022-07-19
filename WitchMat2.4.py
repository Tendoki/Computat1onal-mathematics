import numpy as np
import scipy.linalg
import random as rand
import math


def norma(A):
    m = []
    for i in range(len(A)):
        m.append(sum(abs(A[i])))
    return max(m)


def Jacobi(A, b, eps):

    x = np.transpose(np.matrix([0.0 for i in b]))
    norm = 1
    k = 0
    while norm > eps:
        k += 1
        previos_x = np.copy(x)
        for i in range(len(A)):
            s = 0
            for j in range(len(A)):
                if i != j:
                    s += A[i, j] * x[j]
            previos_x[i] = (b[i] - s) / A[i, i]
        norm = np.sqrt(sum((previos_x[i] - x[i]) ** 2 for i in range(len(A)))) #вторая норма
        x = np.copy(previos_x)
    print('Апосториорная', k, '\n')
    return x


def seidel(A, b, eps):
    n = len(A)
    x = np.transpose(np.matrix([0.0 for i in b]))

    norm = 1
    k = 0
    while norm > eps:
        k += 1
        previos_x = np.copy(x)
        for i in range(n):
            s1 = sum([A[i, j] * previos_x[j] for j in range(i)])
            s2 = sum([A[i, j] * x[j] for j in range(i + 1, n)])
            previos_x[i] = (b[i] - s1 - s2) / A[i, i]

        norm = np.sqrt(sum((previos_x[i] - x[i]) ** 2 for i in range(n)))
        x = previos_x
    print('Апосториорная', k, '\n')
    return x


eps = 0.0000001
n = 4
b = np.transpose(np.matrix([rand.randint(1, 10) for i in range(n)]))
#A = np.matrix([[rand.randint(0, 10) + 0.0 for j in range(n)] for i in range(n)])

# A = (-(2*n))*np.random.sample((n, n)) + n
A = n * np.random.sample((n, n))
for i in range(len(A)):
    A[i, i] = sum([A[i, j] + 5 for j in range(A.shape[1])])

L = np.zeros((n, n))
D = np.zeros((n, n))
R = np.zeros((n, n))

for i in range(A.shape[0]):
    if i == 0:
        D[i, i] = A[i, i]
        R[i, i + 1:] = A[i, i + 1:]
    else:
        L[i, :i] = A[i, :i]
        D[i, i] = A[i, i]
        R[i, i + 1:] = A[i, i + 1:]

print("A=", A)
print("b=", b)

print("Якоби: ")
c = np.dot(np.linalg.inv(D), b)
B = -np.dot(np.linalg.inv(D), (L + R))
x = Jacobi(A, b, eps * (1 - norma(B)) / norma(B))
print('Априорная', np.ceil(math.log(eps * (1 - norma(B)) / norma(c), norma(B))), '\n')
print("x =", x)
print("Зейделя: ")
q = norma(np.dot(np.linalg.inv(L+D), R))

x = seidel(A, b, eps * (1 - q) / q)
print('Априорная', np.ceil(math.log(eps * abs(1 - q) / norma(c), q)), '\n')

print("x =", x)