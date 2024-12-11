import numpy as np
import sympy as sp
import random
import pandas as pd

# Define methods as in your original code
def bisection_method(func, a, b, tol=1e-6, max_iter=100):
    iter_count = 0
    while iter_count < max_iter:
        c = (a + b) / 2.0
        if func(a) * func(c) < 0:
            b = c
        elif func(b) * func(c) < 0:
            a = c
        else:
            break
        if abs(func(c)) < tol or abs(b - a) < tol:
            break
        iter_count += 1
    return c, iter_count, abs(func(c))

def false_position_method(func, a, b, tol=1e-6, max_iter=100):
    iter_count = 0
    c = a
    while iter_count < max_iter:
        c = b - (func(b) * (b - a)) / (func(b) - func(a))
        if func(a) * func(c) < 0:
            b = c
        elif func(b) * func(c) < 0:
            a = c
        else:
            break
        if abs(func(c)) < tol or abs(b - a) < tol:
            break
        iter_count += 1
    return c, iter_count, abs(func(c))

def newton_raphson_method(func, dfunc, x0, tol=1e-6, max_iter=100):
    iter_count = 0
    x = x0
    while iter_count < max_iter:
        if dfunc(x) == 0:
            break
        x_new = x - func(x) / dfunc(x)
        if abs(x_new - x) < tol or abs(func(x_new)) < tol:
            x = x_new
            break
        x = x_new
        iter_count += 1
    return x, iter_count, abs(func(x))

# Generate random equations
def generate_random_equation():
    x = sp.symbols('x')
    degree = random.randint(2, 4)
    coefficients = [random.randint(-10, 10) for _ in range(degree + 1)]
    equation = sum(c * x**i for i, c in enumerate(coefficients))
    return equation

# Train on many equations
training_data = []
x = sp.symbols('x')

for _ in range(100):  # Train with 100 equations
    equation = generate_random_equation()
    f = sp.lambdify(x, equation, "numpy")
    df = sp.lambdify(x, sp.diff(equation, x), "numpy")
    
    # Define interval and initial guess
    a, b = -10, 10
    x0 = random.uniform(-10, 10)
    
    equation_data = {"Equation": str(equation)}
    try:
        equation_data["Bisection"] = bisection_method(f, a, b)[1]  # Store iteration count
    except Exception:
        equation_data["Bisection"] = float('inf')
    
    try:
        equation_data["False Position"] = false_position_method(f, a, b)[1]
    except Exception:
        equation_data["False Position"] = float('inf')
    
    try:
        equation_data["Newton-Raphson"] = newton_raphson_method(f, df, x0)[1]
    except Exception:
        equation_data["Newton-Raphson"] = float('inf')
    
    training_data.append(equation_data)

# Save the training data
df_training = pd.DataFrame(training_data)
df_training.to_csv("numerical_methods_training.csv", index=False)
