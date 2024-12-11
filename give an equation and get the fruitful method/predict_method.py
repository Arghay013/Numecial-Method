import pandas as pd
import sympy as sp
import numpy as np
import random

# Load the training data
def load_training_data(filename):
    try:
        df_training = pd.read_csv(filename)
        print("Training data loaded successfully!")
        return df_training
    except FileNotFoundError:
        print(f"Error: {filename} not found. Make sure the file exists in the same directory.")
        exit()

# Define numerical methods (reuse your previous code)
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

# Predict the best method for a new equation
def predict_best_method(equation):
    x = sp.symbols('x')
    f = sp.lambdify(x, equation, "numpy")
    df = sp.lambdify(x, sp.diff(equation, x), "numpy")
    
    # Define interval and initial guess
    a, b = -10, 10
    x0 = random.uniform(-10, 10)

    # Evaluate each method
    results = {}
    try:
        results["Bisection"] = bisection_method(f, a, b)[1]
    except Exception:
        results["Bisection"] = float('inf')

    try:
        results["False Position"] = false_position_method(f, a, b)[1]
    except Exception:
        results["False Position"] = float('inf')

    try:
        results["Newton-Raphson"] = newton_raphson_method(f, df, x0)[1]
    except Exception:
        results["Newton-Raphson"] = float('inf')

    # Determine the best method
    best_method = min(results, key=results.get)
    return best_method, results

if __name__ == "__main__":
    # Load training data
    filename = "numerical_methods_training.csv"
    df_training = load_training_data(filename)
    
    # Input a new equation
    new_equation_str = input("Enter your equation in terms of x (e.g., x**3 - 6*x + 2): ")
    new_equation = sp.sympify(new_equation_str)

    # Predict the best method
    best_method, all_results = predict_best_method(new_equation)
    print(f"The best method for solving {new_equation} is: {best_method}")
    print(f"Performance details: {all_results}")
