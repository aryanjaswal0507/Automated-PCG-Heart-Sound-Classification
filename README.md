# 🩺 Automated PCG Heart Sound Classification 💓

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest-green.svg)](https://scikit-learn.org/)

A professional machine learning pipeline for the classification of **Phonocardiogram (PCG)** signals. This project leverages advanced signal decomposition techniques and meta-heuristic optimization to achieve high-accuracy detection of heart abnormalities.

---

## 🚀 **Key Features**

*   **Empirical Wavelet Transform (EWT)**: Adaptive decomposition of 1D heart sounds into clinical frequency modes.
*   **Multi-Domain Feature Engine**: Extracts 40+ diagnostic features including Statistical, Spectral Centroid, MFCCs, and Wavelet energy.
*   **Automated Pipeline**: End-to-end workflow from raw data loading to optimized model training.
*   **SMO Optimizer**: Implements a custom **Spider Monkey Optimization** meta-heuristic for hyperparameter tuning.
*   **Robust Data Handling**: Automatic fallback for missing signal data and intelligent label discovery.

---


## 🛠️ **Installation & Usage**

### **1. Setup Environment**
Ensure you have Python 3.10+ installed. Install the required libraries:
```bash
pip install -r requirements.txt
```

### **2. Run the Full Analysis**
Execute the entire pipeline (loading, comparison, and optimization):
```bash
python main.py
```

### **3. Quick Test**
Verify the setup on a small subset of data:
```bash
python main.py --test
```

---

## 📊 **Methodology**

### **Signal Decomposition (EWT)**
We use **Empirical Wavelet Transform** to isolate heart sound components based on their frequency content. This significantly improves the signal-to-noise ratio and highlights pathological transient sounds like murmurs.

### **Classification & Optimization**
The pipeline evaluates **Random Forest**, **SVM**, **KNN**, and **Decision Trees**. The best model is then optimized using a population-based **SMO algorithm**, which mimics social behavior to find the global optimum in the hyperparameter space.

---

## 📜 **License**
Distributed under the MIT License. See `LICENSE` for more information.

## 🤝 **Contributing**
Contributions are welcome! Please open an issue or submit a pull request.
