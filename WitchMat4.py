import math
import numpy as np
import sympy
from sympy import Symbol
from sympy.solvers import solve

a = 0.1
b = 2.3
alpha = 1/5
beta = 0
right_solution = 3.578861536040539915439859609644293194417
speed = 0

def f(x): return 2.5 * sympy.cos(2 * x) * sympy.exp(2 * x / 3) + 4 * sympy.sin(3.5 * x) * sympy.exp(-3 * x) + 3 * x

def omega(x, matrix_a):
    return sum([x ** i * (matrix_a[i] if i < len(matrix_a) else 1) for i in range(len(matrix_a), -1, -1)])

#Моменты аналитическое инегрирование x^i/(x-a)^alpha
mu_0 = lambda x: (x - a) ** (1 - alpha) / (1 - alpha)
mu_1 = lambda x: -(x - a) ** (1 - alpha) * ((alpha - 1) * x - a) / ((alpha - 2) * (alpha - 1))
mu_2 = lambda x: -((x-a) ** (1-alpha) * ((alpha-2)*(alpha-1)*x**2-2*a*(alpha-1)*x+2*a**2))/((alpha-3)*(alpha-2)*(alpha-1))
mu_3 = lambda x: -((x-a)**(1-alpha)*((alpha-3)*(alpha-2)*(alpha-1)*x**3-3*a*(alpha-2)*(alpha-1)*x**2+6*a**2*(alpha-1)*x-6*a**3))/((alpha-4)*(alpha-3)*(alpha-2)*(alpha-1))
mu_4 = lambda x: (x-a)**(5-alpha)/(5-alpha)+(4*a*(x-a)**(4-alpha))/(4-alpha)+(6*a**2*(x-a)**(3-alpha))/(3-alpha)+(4*a**3*(x-a)**(2-alpha))/(2-alpha)+(a**4*(x-a)**(1-alpha))/(1-alpha)
mu_5 = lambda x: (x-a)**(6-alpha)/(6-alpha)+(5*a*(x-a)**(5-alpha))/(5-alpha)+(10*a**2*(x-a)**(4-alpha))/(4-alpha)+(10*a**3*(x-a)**(3-alpha))/(3-alpha)+(5*a**4*(x-a)**(2-alpha))/(2-alpha)+(a**5*(x-a)**(1-alpha))/(1-alpha)

def newton_cotes(a, b, show=False):
    node = np.array([a, (a + b) / 2, b])
    moments = [mu_0(b) - mu_0(a), mu_1(b) - mu_1(a), mu_2(b) - mu_2(a)] #По Ньютону-Лейбница
    if show: print("Моменты: ", moments)
    mu_s = np.transpose([np.array(moments, dtype="float")])
    x_s = np.array([node ** 0, node, node ** 2], dtype="float")
    if show: print("x_i матрица: \n", x_s, '\n')

    A = np.linalg.solve(x_s, mu_s)
    if show: print("A матрица:\n", A, '\n')

    solution = sum([f(node[i]) * A[i] for i in range(3)])
    if show: print("Решение:", solution, '\n')

    M_n = 211.1  # максимум |d3/dx3 f(x)|
    integral = 0.0765419323739311  #|p(x)w(x)| от a до b
    R_n = M_n / 6 * integral
    if show: print("Оценка погрешности:", R_n)
    if show: print("Точная Погрешность:", right_solution - solution)
    return solution

def gauss(a, b, show=False):
    moments = [mu_0(b) - mu_0(a), mu_1(b) - mu_1(a), mu_2(b) - mu_2(a),
               mu_3(b) - mu_3(a), mu_4(b) - mu_4(a), mu_5(b) - mu_5(a)]
    if show: print("Моменты: ", moments, '\n')
    mu_js = np.array(
        [[moments[0], moments[1], moments[2]],
         [moments[1], moments[2], moments[3]],
         [moments[2], moments[3], moments[4]]])
    mu_ns = -np.array([[moments[3]], [moments[4]], [moments[5]]])
    matrix_a = np.linalg.solve(mu_js, mu_ns)
    if show:
        print("Коэффициенты моменты j+s: \n", mu_js, '\n')
        print("Моменты n+s: \n", mu_ns, '\n')
        print("Матрица a: \n", matrix_a, '\n')

    x = Symbol('x')
    matrix_x = solve(omega(x, matrix_a))
    matrix_x = np.array([value.get(x).args[0] for value in matrix_x])
    if show: print("Матрица x_j:\n", matrix_x, '\n')

    x_s = np.array([matrix_x ** 0, matrix_x ** 1, matrix_x ** 2], dtype="float")
    mu_s = np.array([[moments[0]], [moments[1]], [moments[2]]])
    A = np.linalg.solve(x_s, mu_s)
    if show: print("A матрица:\n", A, '\n')

    solution = sum([f(matrix_x[i]) * A[i] for i in range(3)])
    if show: print("Решение:", solution, '\n')

    M_n = 36145.4  # максимум |d6/dx6 f(x)|
    integral = 0.0308842677882016  #|p(x)w^2(x)| от a до b
    if show:
        print("Оценка погрешности: ", M_n / math.factorial(6) * integral)
        print("Точная Погрешность: ", right_solution - solution)
    return solution

def SKF(a, b, typeBuild, k=1):
    solutions = []
    solution = 0
    h_r = []
    while abs(right_solution - solution) > 1e-6:
        h = (b - a) / k
        h_r.append(h)
        lb = a
        ub = a + h
        solution = 0
        step = 1
        while ub <= b:
            solution += typeBuild(lb, ub)
            lb = a + h * step
            ub = a + h * (step + 1)
            step += 1
        print("\nРешение", step-1, ": ", solution)
        solutions.append(solution)
        k *= 2

        global speed
        if len(solutions) >= 3:
            s = len(solutions)
            speed = -(math.log(abs((solutions[s - 1] - solutions[s - 2]) / (solutions[s - 2] - solutions[s - 3]))) / math.log(2))
            print("Скорость:", speed, end='')
    solutions = np.array(solutions, dtype='float')
    m = math.floor(speed)
    h_mi = []
    for i in range(len(h_r)):
        h_mi.append([1])
        h_mi[i].extend([-(h_r[i]**j) for j in range(m, m+len(h_r)-1)])
    h_mi = np.array(h_mi, dtype='float')
    C_p = np.linalg.solve(h_mi, solutions)
    delta = np.subtract(C_p[0], solutions[-1])
    print("\nОценка погрешности по Ричардсону:", delta)
    return solutions

print("Ньютон-Котс")
newton_cotes(a, b, True)
print("СКФ Ньютон-Котс")
solutions = SKF(a, b, newton_cotes)
print("СКФ Ньютон-Котс h_opt")
Rh_1 = (solutions[1] - solutions[0]) / (1 - 2 ** (-speed))
h_opt = (b - a) / 2 * ((1e-6 / abs(Rh_1)) ** (1 / speed)) * 0.95
print("h-opt ", h_opt)
SKF(a, b, newton_cotes, math.ceil((b - a) / h_opt))
print("Гаусс")
gauss(a, b, True)
print("СКФ Гаусс")
solutions = SKF(a, b, gauss)
print("СКФ Гаусс h_opt")
Rh_1 = (solutions[1] - solutions[0]) / (1 - 2 ** (-speed))
h_opt = (b - a) / 2 * ((1e-6 / abs(Rh_1)) ** (1 / speed)) * 0.95
print("h-opt ", h_opt)
SKF(a, b, gauss, math.ceil((b - a) / h_opt))