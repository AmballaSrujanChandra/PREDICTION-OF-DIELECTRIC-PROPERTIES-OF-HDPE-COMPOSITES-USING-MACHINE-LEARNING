# **Prediction of Dielectric Properties of HDPE Composites using Machine Learning**

## 📌 **Project Overview**

This project focuses on predicting the **dielectric constant** and **dielectric loss** of **HDPE (High-Density Polyethylene) + TiO₂ composites** using various Machine Learning algorithms. The aim is to accelerate the material development process by minimizing the need for extensive experimental testing through data-driven approaches.

## 🎯 **Objective**

- Develop a straightforward ML-based approach to rapidly predict **dielectric permittivity** and **dielectric loss**
- Compare the performance of different algorithms and identify the **best fit** for this dataset
- Analyze limitations of ML in **extrapolating** beyond the trained data range

## 🧩 **Dataset**

### **Features:**
- **Percentage of HDPE**
- **Frequency**

### **Target Variables:**
- **Dielectric Constant (K)**
- **Dielectric Loss (D)**

Data includes multiple TiO₂ composite concentrations (**H0T, H10T, H20T, H30T, H40T**).

## ⚙️ **Methodology**

1. **Data Collection & Cleaning**
2. **Feature Engineering**
3. **Model Building**
4. **Validation & Testing**
5. **Visualization of Results**

### **Implemented Models**
- **Random Forest**
- **Gradient Boosting**
- **Decision Tree**
- **AdaBoost**
- **KNN**
- **Neural Networks**
- **SVM**

## 📊 **Results**

### **Training/Test Split (80/20):**
- **Random Forest** performed the best with:
  - **R² = 0.9922** (Dielectric Loss)
  - **R² = 0.9998** (Dielectric Constant)
- **Gradient Boosting** also performed well (**R² ≈ 0.988 / 0.9998**)
- **KNN** performed poorly, showing negative R² values

### **Cross-composition Testing** (e.g., training without H30T and testing on H30T):
- Most models gave **negative results** due to poor extrapolation beyond trained data ranges

## 🚀 **Key Insights**

- **Random Forest** is the most reliable algorithm for this dataset
- **Machine Learning models struggle with extrapolation**—predictions are accurate only within the range of trained data
- Ensuring **representative, high-quality datasets** is critical for reliable predictions

## 📂 **Repository Structure**
```
├── Decision Tree.py
├── Gradient Boosting.py
├── KNN.py
├── Neural Networks.py
├── Random Forest.py
├── SVM.py
├── adaboost.py
├── Predicting Dielectric constant and dielectric loss of HDPE.pptx
└── README.md
```

## 🛠️ **Requirements**

- **Python 3.x**
- **pandas**
- **numpy**
- **scikit-learn**
- **matplotlib**

### **Install dependencies with:**
```bash
pip install -r requirements.txt
```

## ▶️ **Usage**

Run any model script directly, for example:

```bash
python Random Forest.py
```

## 📌 **Future Work**

- **Expand dataset** with broader compositions
- **Explore deep learning models** with better generalization
- **Investigate physics-informed ML approaches** for improved extrapolation