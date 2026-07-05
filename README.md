# 🌸 Iris Flower Species Classifier

<div align="center">

**A two-tier ML web application — Streamlit frontend + Flask API backend — for real‑time Iris species prediction**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white&style=for-the-badge)
![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white&style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white&style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white&style=for-the-badge)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white&style=for-the-badge)
![RandomForest](https://img.shields.io/badge/Model-RandomForest-success?style=for-the-badge)

</div>

---

## 📑 Table of Contents

| # | Section |
|---|---------|
| 1 | [🧠 Overview](#-overview) |
| 2 | [🏗️ Architecture](#-architecture) |
| 3 | [📁 Project Structure](#-project-structure) |
| 4 | [✨ Features](#-features) |
| 5 | [🛠️ Installation](#-installation) |
| 6 | [🚀 Usage](#-usage) |
| 7 | [📊 Data](#-data) |
| 8 | [🤖 Model](#-model) |
| 9 | [🎓 Training Pipeline](#-training-pipeline) |
| 10 | [🔌 API Reference](#-api-reference) |
| 11 | [📈 Evaluation](#-evaluation) |
| 12 | [📜 License](#-license) |

---

## 🧠 Overview

This project implements a machine learning pipeline that trains a classifier on the classic Iris dataset and exposes it through a modern two‑tier web architecture:

| Tier | Technology | Role |
|------|------------|------|
| 🎨 **Frontend** | Streamlit + Plotly | User interface with sliders & probability charts |
| ⚙️ **Backend** | Flask REST API | Serves model predictions via `/predict` |
| 🧠 **Model** | Random Forest (scikit-learn) | Classifies flowers into 3 species |

**Prediction flow:** User adjusts sliders → Streamlit sends JSON to Flask → Flask runs inference → Probability bar chart rendered

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Two‑Tier ML Web Application                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────┐        ┌──────────────────────────┐  │
│  │   🎨 Streamlit UI    │  POST  │    ⚙️ Flask API :5000    │  │
│  │   (app.py)           │───────▶│    (flask_api.py)        │  │
│  │                      │ JSON   │                          │  │
│  │  ┌────────────────┐  │features│  ┌────────────────────┐  │  │
│  │  │ Sepal sliders  │  │        │  │ RandomForest       │  │  │
│  │  │ Petal sliders  │  │        │  │ model.pkl          │  │  │
│  │  │ (4 inputs)     │  │        │  │ (191 KB)           │  │  │
│  │  └────────────────┘  │        │  └────────────────────┘  │  │
│  │         │             │        │           │               │  │
│  │         ▼             │        │           ▼               │  │
│  │  ┌────────────────┐  │◀───────│  { label, probabilities } │  │
│  │  │ Plotly Bar     │  │  JSON  └──────────────────────────┘  │
│  │  │ Chart          │  │                                      │
│  │  └────────────────┘  │                                      │
│  └──────────────────────┘                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Iris-Model/
├── README.md               ← You are here
├── requirement.txt         ← Python dependencies (8 packages)
├── app.py                  ← Streamlit frontend (UI + Plotly charts)
├── flask_api.py            ← Flask REST backend (serves model)
├── train.py                ← Model training script
├── data/
│   └── Iris.csv            ← Dataset (150 samples × 5 columns)
└── model/
    └── model.pkl           ← Serialized RandomForest model (191 KB)
```

> ⚠️ **Note:** `train.py` saves the model to `model/random_forest.pkl` while `flask_api.py` loads from `model/model.pkl`. Rename the output file after training or update `flask_api.py` to match.

---

## ✨ Features

| Feature | Detail |
|---------|--------|
| 🎚️ **Interactive Sliders** | Adjust sepal/petal measurements via Streamlit sliders |
| 🔗 **REST API** | Clean JSON contract between frontend & backend |
| 📊 **Probability Visualization** | Plotly bar chart with per‑class confidence scores |
| 🌐 **Separation of Concerns** | UI (`app.py`) and inference (`flask_api.py`) run independently |
| 🧠 **Ensemble Model** | Random Forest — high accuracy, robust to overfitting |
| 🎨 **Responsive Layout** | Two‑column slider grid with expandable info panel |

---

## 🛠️ Installation

### Step 1 — Clone & Enter

```bash
git clone <repo-url>
cd Iris-Model
```

### Step 2 — Install Dependencies

```bash
pip install -r requirement.txt
```

### 📦 Dependency Breakdown

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI framework |
| `flask` | REST API server |
| `scikit-learn` | Model training & inference |
| `plotly` | Interactive bar charts |
| `pandas` | Data loading & manipulation |
| `numpy` | Numerical operations |
| `requests` | HTTP client (frontend → backend calls) |
| `matplotlib` / `seaborn` | Exploratory data analysis & training visuals |

---

## 🚀 Usage

### Terminal 1 — Start the Flask API

```bash
python flask_api.py
```

```
 * Serving Flask app 'flask_api'
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

### Terminal 2 — Launch the Streamlit App

```bash
streamlit run app.py
```

### Browser — Predict

Open **http://localhost:8501**. Use the sliders to set flower measurements, and the predicted species + probability chart appears instantly.

---

## 📊 Data

**Source:** Classic Iris dataset (Fisher, 1936 — UCI Machine Learning Repository)

| Property | Value |
|----------|-------|
| Total samples | **150** |
| Features | Sepal length, Sepal width, Petal length, Petal width |
| Classes | **Setosa** (0), **Versicolor** (1), **Virginica** (2) |
| Class balance | 50 samples per species (perfectly balanced) |
| File | `data/Iris.csv` |

**Label encoding** (in `train.py`):
```python
data['species'] = data['species'].map({
    'setosa': 0,
    'versicolor': 1,
    'virginica': 2
})
```

---

## 🤖 Model

| Attribute | Value |
|-----------|-------|
| **Algorithm** | Random Forest Classifier |
| **Library** | scikit-learn `RandomForestClassifier()` |
| **Default params** | 100 trees, Gini impurity, no max depth |
| **Serialized file** | `model/model.pkl` (191 KB) |
| **Train/test split** | 80% / 20% (`random_state=42`) |

---

## 🎓 Training Pipeline

```bash
python train.py
```

```
          Load               Split           Train              Evaluate
            │                  │                │                   │
Iris.csv ──▶ value_counts()   │                │                   │
            │                  │                │                   │
            ▼                  ▼                ▼                   ▼
       label encode      80/20 split     RandomForest.fit()   accuracy_score()
       (setosa→0)        (seed=42)           model.predict()     ypred print
            │                  │                │                   │
            │                  │                ▼                   │
            │                  │        joblib.dump(model,
            │                  │        'model/random_forest.pkl')
```

**Training output:**
1. Prints class distribution (`data['species'].value_counts()`)
2. Prints predicted labels on the test set
3. Prints accuracy score (typically **~95–97%**)

---

## 🔌 API Reference

### `GET /`

Returns a welcome message.

```
"Welcome to the Iris Flower Prediction API!"
```

### `POST /predict`

**Request:**
```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

Feature order: `[sepal_length, sepal_width, petal_length, petal_width]`

**Success Response (200):**
```json
{
  "prediction": 0,
  "label": "setosa",
  "probabilities": {
    "setosa": 0.97,
    "versicolor": 0.02,
    "virginica": 0.01
  }
}
```

**Error Response:**
```json
{ "error": "<error description>" }
```

---

## 📈 Evaluation

<div align="center">

| Metric | Value |
|--------|-------|
| **Model** | Random Forest (default hyperparams) |
| **Accuracy** | ~95–97% on 20% hold‑out set |
| **Split seed** | `random_state=42` |
| **Classes** | 3 (balanced: 10/test each) |

</div>

> Run `python train.py` to reproduce metrics — accuracy is printed at the end of training.

---

## 📜 License

<div align="center">

![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

This project is open‑source under the **MIT License**.

</div>

---

<div align="center">

### ⭐ Found this useful? Give it a star!

**Built with ❤️ using Python, Flask, Streamlit, and scikit-learn**

</div>