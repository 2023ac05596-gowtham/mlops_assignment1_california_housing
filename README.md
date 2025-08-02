# 🏠 MLOps Assignment 1 — California Housing Price Prediction

This repository contains a complete MLOps pipeline for predicting California housing prices using the `California Housing` dataset.

---

## 📌 Objective

Implement a reproducible and trackable machine learning workflow using:
- **DVC** for data and code versioning
- **MLflow** for experiment tracking and model registry
- **Git** for source control

---

## 📂 Project Structure

```
.
├── data/
│   └── california_housing.csv         # Raw dataset (tracked by DVC)
├── src/
│   └── train_model.py                 # Training script with MLflow tracking
├── dvc.yaml                           # DVC pipeline config
├── .gitignore                         # Git ignore rules
├── .dvcignore                         # DVC ignore rules
├── README.md                          # This file
```

---

## 🔧 Setup Instructions

### 1. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Reproduce data and training pipeline

```bash
dvc repro
```

### 4. Run training manually

```bash
python src/train_model.py
```

This will:
- Train **Linear Regression** and **Decision Tree** models.
- Track metrics (RMSE, R²) in **MLflow**.
- Register both models in the MLflow **Model Registry**.

---

## 📊 Experiment Tracking

Run the following to launch MLflow UI:

```bash
mlflow ui
```

Then open [http://localhost:5000](http://localhost:5000) to view experiments and models.

---

## ✅ Completed Milestones

### ✅ Part 1: Versioning
- [x] Raw data versioned with **DVC**
- [x] Project code versioned with **Git**

### ✅ Part 2: Model Development & Experiment Tracking
- [x] Trained 2 models
- [x] Metrics tracked in **MLflow**
- [x] Best model registered in Model Registry

---

## 📌 Upcoming (Part 3+)
- Model packaging with Flask/FastAPI
- Docker containerization
- CI/CD with GitHub Actions
- Logging & Monitoring

---

## 👨‍💻 Author
**Gowtham Raj R.**