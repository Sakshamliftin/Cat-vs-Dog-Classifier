# Optimization Algorithms - Comprehensive Implementation

This notebook provides a complete implementation of 17 optimization algorithms with detailed visualizations and explanations.

## 📚 Algorithms Implemented

### First-Order Methods
1. **Basic Gradient Descent** - Foundation of iterative optimization
2. **Exact Line Search** - Optimal step size selection
3. **Armijo Rule (Backtracking)** - Adaptive step size with sufficient decrease
4. **Stochastic Gradient Descent (SGD)** - Sample-by-sample updates
5. **Mini-Batch Gradient Descent** - Batch-based efficient updates
6. **EWMA Momentum** - Accelerated gradient descent with momentum

### Application-Specific Methods
7. **Linear Regression (GD)** - Least squares optimization
8. **Logistic Regression** - Binary classification with cross-entropy

### Non-Smooth and Regularized Methods
9. **Subgradient Descent** - Optimization for non-differentiable functions
10. **L1 Regularization (Lasso)** - Sparse solution recovery
11. **L2 Regularization (Ridge)** - Smooth coefficient shrinkage

### Second-Order Methods
12. **Newton's Method (Root Finding)** - Fast root finding with quadratic convergence
13. **Newton's Method (Optimization)** - Hessian-based optimization

### Constrained Optimization
14. **Lagrange Multiplier Method** - Equality-constrained optimization
15. **KKT Conditions** - General constrained optimization framework
16. **Active Set Method** - Inequality-constrained optimization

## 🎯 Features

Each algorithm includes:
- ✅ **Mathematical Formulation** - Clear equations and update rules
- ✅ **Clean Python Implementation** - Well-documented, readable code
- ✅ **Comprehensive Visualizations** - Convergence plots, optimization paths, contour plots
- ✅ **Real Example Datasets** - Practical demonstrations
- ✅ **Detailed Explanations** - Understanding the intuition and theory

## 📊 Visualizations

The notebook includes:
- Convergence plots showing iteration vs objective value
- Optimization path visualization on contour plots
- Parameter evolution tracking
- Comparative analysis between methods
- Decision boundaries for classification
- Constraint and feasible region visualization
- Active constraint identification

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scipy jupyter notebook
```

### Running the Notebook
```bash
cd Notebook
jupyter notebook optimization_algorithms.ipynb
```

## 📖 Algorithm Summary

| Algorithm | Type | Convergence | Key Feature |
|-----------|------|-------------|-------------|
| Gradient Descent | Unconstrained | Linear | Fixed step size |
| Exact Line Search | Unconstrained | Linear | Optimal step size |
| Armijo Backtracking | Unconstrained | Linear | Adaptive step size |
| Linear Regression | Unconstrained | Linear | Least squares |
| SGD | Unconstrained | Sublinear | Sample-by-sample |
| Mini-Batch GD | Unconstrained | Linear | Batch updates |
| Logistic Regression | Unconstrained | Linear | Binary classification |
| EWMA Momentum | Unconstrained | Faster | Accelerated |
| Subgradient | Non-smooth | Sublinear | Non-differentiable |
| Lasso (L1) | Regularized | Linear | Sparse solutions |
| Ridge (L2) | Regularized | Linear | Smooth shrinkage |
| Newton (Root) | Root Finding | Quadratic | Very fast near root |
| Newton (Opt) | Second-order | Quadratic | Uses Hessian |
| Lagrange | Equality Const. | Problem-dependent | Equality constraints |
| KKT | Inequality Const. | Problem-dependent | General constraints |
| Active Set | Inequality Const. | Finite (QP) | Identifies active set |

## 🎓 Learning Objectives

After working through this notebook, you will understand:
- How gradient-based optimization works
- Trade-offs between different optimization methods
- When to use first-order vs second-order methods
- How to handle constraints in optimization
- Regularization techniques for better generalization
- Practical implementation details and best practices

## 🔬 Example Problems

The notebook demonstrates algorithms on:
- **Rosenbrock Function** - Classic non-convex test function
- **Quadratic Functions** - Simple convex optimization
- **Linear Regression** - Real-world data fitting
- **Logistic Regression** - Binary classification
- **Sparse Recovery** - High-dimensional problems
- **Constrained Problems** - Feasibility and optimality

## 📝 Notes

- All implementations use NumPy for efficient computation
- Visualizations use Matplotlib and Seaborn
- Random seeds are set for reproducibility
- Code is optimized for clarity over performance
- Each algorithm is self-contained and can be run independently

## 🤝 Contributing

This notebook is part of the Cat-vs-Dog-Classifier project. Feel free to:
- Add more test functions
- Implement additional algorithms
- Improve visualizations
- Add more detailed explanations

## 📚 References

The implementations are based on standard optimization textbooks and papers:
- Nocedal & Wright: Numerical Optimization
- Boyd & Vandenberghe: Convex Optimization
- Goodfellow et al.: Deep Learning

## ⚖️ License

This notebook is provided for educational purposes.
