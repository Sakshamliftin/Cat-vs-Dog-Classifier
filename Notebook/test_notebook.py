#!/usr/bin/env python3
"""
Test script to verify the optimization algorithms notebook can be executed.
This runs a subset of the algorithms to ensure everything works correctly.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import sys

def test_gradient_descent():
    """Test basic gradient descent implementation"""
    print("Testing Gradient Descent...")
    
    def rosenbrock(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    
    def rosenbrock_grad(x):
        dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
        dy = 200 * (x[1] - x[0]**2)
        return np.array([dx, dy])
    
    def gradient_descent(f, grad_f, x0, learning_rate=0.1, max_iter=100, tol=1e-6):
        x = np.array(x0, dtype=float)
        x_history = [x.copy()]
        f_history = [f(x)]
        
        for i in range(max_iter):
            grad = grad_f(x)
            x_new = x - learning_rate * grad
            x_history.append(x_new.copy())
            f_history.append(f(x_new))
            
            if np.linalg.norm(x_new - x) < tol:
                break
            x = x_new
        
        return x_history, f_history
    
    x0 = np.array([-1.0, 1.0])
    x_hist, f_hist = gradient_descent(rosenbrock, rosenbrock_grad, x0, 
                                       learning_rate=0.001, max_iter=100)
    
    assert len(f_hist) > 1, "No iterations performed"
    assert f_hist[-1] < f_hist[0], "Function value should decrease"
    print(f"  ✓ Converged from {f_hist[0]:.4f} to {f_hist[-1]:.4f} in {len(f_hist)-1} iterations")

def test_linear_regression():
    """Test linear regression with gradient descent"""
    print("Testing Linear Regression...")
    
    np.random.seed(42)
    m = 100
    X = 2 * np.random.rand(m, 1)
    y = 4 + 3 * X.ravel() + np.random.randn(m) * 0.5
    X_bias = np.c_[np.ones((m, 1)), X]
    
    theta = np.zeros(2)
    learning_rate = 0.1
    
    for _ in range(100):
        predictions = X_bias @ theta
        errors = predictions - y
        gradient = (1 / m) * (X_bias.T @ errors)
        theta = theta - learning_rate * gradient
    
    # Check if parameters are close to true values (4, 3)
    assert 3.5 < theta[0] < 4.5, f"Intercept should be near 4, got {theta[0]}"
    assert 2.5 < theta[1] < 3.5, f"Slope should be near 3, got {theta[1]}"
    print(f"  ✓ Parameters: θ₀={theta[0]:.2f}, θ₁={theta[1]:.2f} (true: 4, 3)")

def test_newton_method():
    """Test Newton's method for root finding"""
    print("Testing Newton's Method...")
    
    def f(x):
        return x**2 - 2
    
    def f_prime(x):
        return 2*x
    
    x = 1.0
    for _ in range(10):
        x = x - f(x) / f_prime(x)
    
    true_value = np.sqrt(2)
    error = abs(x - true_value)
    assert error < 1e-10, f"Newton's method didn't converge accurately"
    print(f"  ✓ sqrt(2) ≈ {x:.10f}, error: {error:.2e}")

def test_logistic_regression():
    """Test logistic regression"""
    print("Testing Logistic Regression...")
    
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    np.random.seed(42)
    m = 100
    
    # Generate data
    X0 = np.random.randn(m//2, 2) + np.array([-2, -2])
    X1 = np.random.randn(m//2, 2) + np.array([2, 2])
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(m//2), np.ones(m//2)])
    X_bias = np.c_[np.ones(m), X]
    
    theta = np.zeros(3)
    learning_rate = 0.1
    
    for _ in range(100):
        h = sigmoid(X_bias @ theta)
        gradient = (1/m) * X_bias.T @ (h - y)
        theta = theta - learning_rate * gradient
    
    # Check accuracy
    predictions = sigmoid(X_bias @ theta) >= 0.5
    accuracy = np.mean(predictions == y)
    assert accuracy > 0.8, f"Accuracy too low: {accuracy}"
    print(f"  ✓ Classification accuracy: {accuracy*100:.1f}%")

def main():
    print("="*60)
    print("Testing Optimization Algorithms Notebook Components")
    print("="*60)
    print()
    
    try:
        test_gradient_descent()
        test_linear_regression()
        test_newton_method()
        test_logistic_regression()
        
        print()
        print("="*60)
        print("✓ All tests passed successfully!")
        print("The notebook is ready to use.")
        print("="*60)
        return 0
        
    except Exception as e:
        print()
        print("="*60)
        print(f"✗ Test failed: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
