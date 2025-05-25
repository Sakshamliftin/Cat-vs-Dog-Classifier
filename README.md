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
