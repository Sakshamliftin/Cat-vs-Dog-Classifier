# 🐶🐱 Cat vs Dog Image Classification

This project tackles the classic binary image classification problem — distinguishing between **cats** and **dogs** — using deep learning.

## 🚀 What I Did

- **Built a custom Convolutional Neural Network (CNN)** from scratch with ~**1.4 million trainable parameters**, achieving over **93% test accuracy**.
- Applied **data augmentation** techniques like random zoom, shear, and flips to improve generalization.
- Optimized training with **Dropout layers** to reduce overfitting.
- Scaled and resized images to **128×128** for faster training without compromising accuracy.
- **Implemented Transfer Learning using VGG16**, significantly reducing training time while improving performance.
- Saved the trained model (`.h5`) and built a **Streamlit web app** to allow image uploads and real-time predictions.
- Organized the project for clarity, separating the model, notebook, dataset, and app.

## ✅ Highlights

- ✅ Custom CNN from scratch (~1.4M params)
- ✅ Transfer learning with VGG16
- ✅ ~99% training accuracy, ~93% test accuracy
- ✅ Streamlit frontend for real-time prediction
- ✅ Strong generalization with dropout and augmentation

## 🛠 Tech Used

- TensorFlow & Keras
- NumPy & PIL
- Streamlit for deployment-ready UI

## 📊 Optimization Algorithms Notebook

In addition to the Cat vs Dog classifier, this repository now includes a comprehensive implementation of **17 optimization algorithms** with detailed visualizations and explanations.

### 🎯 Available in: `Notebook/optimization_algorithms.ipynb`

The notebook covers:
- **First-Order Methods**: Gradient Descent, Line Search, Armijo Backtracking, SGD, Mini-Batch GD, Momentum
- **Regression & Classification**: Linear Regression, Logistic Regression
- **Regularization**: Lasso (L1), Ridge (L2), Subgradient Descent
- **Second-Order Methods**: Newton's Method (Root Finding & Optimization)
- **Constrained Optimization**: Lagrange Multipliers, KKT Conditions, Active Set Method

Each algorithm includes:
- Mathematical formulation and theory
- Clean, documented Python implementation
- Comprehensive visualizations (convergence plots, optimization paths, contour plots)
- Real example datasets and practical demonstrations

### �� Documentation
See `Notebook/README_OPTIMIZATION.md` for detailed information about each algorithm.

### 🚀 Quick Start
```bash
cd Notebook
jupyter notebook optimization_algorithms.ipynb
```

### 🧪 Testing
Run the test suite to verify all algorithms:
```bash
python3 Notebook/test_notebook.py
```
