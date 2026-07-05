# 🌸 Iris Species Classifier

<div align="center">

**A complete MLOps pipeline for classifying Iris flower species with interactive visualization**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white&style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-green?logo=scikit-learn&logoColor=white&style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?logo=streamlit&logoColor=white&style=for-the-badge)
![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-blue?logo=plotly&logoColor=white&style=for-the-badge)

![License](https://img.shields.io/github/license/your-org/iris-model?style=flat-square)
![Last Commit](https://img.shields.io/github/last-commit/your-org/iris-model?style=flat-square)
![Issues](https://img.shields.io/github/issues/your-org/iris-model?style=flat-square)

</div>

---

## 📑 Table of Contents

| Section | Description |
|---------|-------------|
| [🚀 Overview](#-overview) | What this project does |
| [✨ Features](#-features) | Key capabilities |
| [🏗️ Architecture](#-architecture) | System design & components |
| [⚙️ Prerequisites](#-prerequisites) | Required software & versions |
| [🛠️ Installation](#-installation) | Setup instructions |
| [📊 Data](#-data) | Dataset information |
| [🤖 Model](#-model) | Model details |
| [🎓 Training](#-training) | How to train the model |
| [📈 Evaluation](#-evaluation) | Performance metrics |
| [🖥️ Demo](#-demo) | Running the application |
| [⚙️ Configuration](#-configuration) | Config options |
| [📁 Project Structure](#-project-structure) | File organization |
| [🤝 Contributing](#-contributing) | How to contribute |
| [📜 License](#-license) | Licensing info |

---

## 🚀 Overview

This repository contains a complete **MLOps pipeline** for:

| Component | Technology | Purpose |
|-----------|------------|---------|
| 🧠 **Model Training** | scikit-learn SVM | Classify Iris species with 96%+ accuracy |
| 🌐 **Web Interface** | Streamlit | Interactive prediction UI |
| 📊 **Visualization** | Plotly | Real-time probability charts |
| 🐳 **Deployment** | Docker | Containerized, reproducible environments |
| 🔧 **Config** | YAML | Centralized settings management |

---

## ✨ Features

<div align="center">

| Feature | Benefit |
|---------|---------|
| 🎯 **High Accuracy** | ~96% test accuracy with RBF kernel SVM |
| 📈 **Interactive Charts** | Plotly bar graphs show prediction confidence |
| 🖱️ **User-Friendly UI** | Simple form input with instant results |
| 🔄 **Batch API** | REST endpoint for multiple predictions |
| 🐳 **One-Click Deploy** | Docker Compose for instant setup |
| ⚙️ **Flexible Config** | YAML + environment variable overrides |

</div>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Iris Species Classifier                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                  │
│  │   Data   │───▶│  Train   │───▶│  Model   │                  │
│  │ (Iris)   │    │  Script  │    │  (.joblib)│                 │
│  └──────────┘    └──────────┘    └────┬─────┘                  │
│                                       │                          │
│                                       ▼                          │
│  ┌──────────────────────────────────────────────────┐          │
│  │              Streamlit Web App                   │          │
│  │  ┌────────────┐  ┌────────────┐  ┌───────────┐  │          │
│  │  │   Input    │──│  Predict   │──│  Plotly   │  │          │
│  │  │  Form      │  │  Engine    │  │  Charts   │  │          │
│  │  └────────────┘  └────────────┘  └───────────┘  │          │
│  └──────────────────────────────────────────────────┘          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| ![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white&style=flat) | 3.10+ | Runtime |
| ![pip](https://img.shields.io/badge/pip-23.0%2B-green?logo=python&logoColor=white&style=flat) | 23.0+ | Package manager |
| ![Docker](https://img.shields.io/badge/Docker-24.0%2B-blue?logo=docker&logoColor=white&style=flat) | 24.0+ | Containerization |
| ![git](https://img.shields.io/badge/git-2.30%2B-orange?logo=git&logoColor=white&style=flat) | 2.30+ | Version control |

---

## 🛠️ Installation

### Quick Start

```bash
# 📥 Clone the repository
git clone https://github.com/your-org/iris-model.git
cd iris-model

# 🪟 Create virtual environment
python -m venv venv

# 🍎/🐧 Activate environment
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 📦 Install dependencies
pip install -r requirements.txt
```

### 🐳 Docker Deployment (Recommended)

```bash
# Build and run
docker-compose up --build

# 🌐 Access at: http://localhost:8501
```

---

## 📊 Data

### 📋 Dataset Overview

| Property | Value |
|----------|-------|
| **Source** | UCI Machine Learning Repository |
| **Samples** | 150 |
| **Features** | 4 (sepal length/width, petal length/width) |
| **Classes** | 3 (Setosa, Versicolor, Virginica) |
| **File** | `data/iris.csv` |

### 📁 Data Structure

```
data/
├── iris.csv          # Raw dataset
└── processed/        # Preprocessed features (optional)
```

---

## 🤖 Model

### 🧠 Model Specifications

| Attribute | Details |
|-----------|---------|
| **Algorithm** | Support Vector Machine (SVM) |
| **Kernel** | RBF (Radial Basis Function) |
| **File** | `models/iris_svm.joblib` |
| **Metadata** | `models/model_metadata.json` |
| **Training Script** | `train.py` |

---

## 🎓 Training

```bash
python train.py
```

### 📊 Training Pipeline

```
1️⃣ Load Data ──▶ 2️⃣ Preprocess ──▶ 3️⃣ Split (80/20) ──▶ 4️⃣ Train SVM ──▶ 5️⃣ Evaluate ──▶ 6️⃣ Save Model
```

**What happens:**
1. 📥 Load and preprocess the Iris dataset
2. 📊 Split into train (80%) and test (20%) sets
3. 🤖 Fit SVM classifier with RBF kernel
4. 📈 Compute metrics (accuracy, precision, recall, F1)
5. 💾 Save model and metadata to disk

---

## 📈 Evaluation

### 🎯 Performance Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | ~96% |
| **Precision** | Per-class detailed report |
| **Recall** | Per-class detailed report |
| **F1-Score** | Per-class detailed report |
| **Confusion Matrix** | Visualized in reports |

Results stored in: `reports/evaluation.json`

---

## 🖥️ Demo

### 🚀 Running the Streamlit App

```bash
streamlit run app.py
```

### 📱 What You Get

| Feature | Description |
|---------|-------------|
| 📝 **Input Form** | Enter sepal/petal dimensions |
| 🎯 **Instant Prediction** | Real-time species classification |
| 📊 **Probability Chart** | Plotly bar graph showing confidence |
| 🎨 **Interactive** | Hover for details, zoom to explore |

<div align="center">

**Expected Output:**
```
Predicted Species: **Iris-setosa**
```
With interactive probability visualization!

</div>

---

## ⚙️ Configuration

### 📄 config.yaml

```yaml
model_path: "models/iris_svm.joblib"
app_port: 8501
debug: false
```

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Path to trained model | `models/iris_svm.joblib` |
| `PORT` | App server port | `8501` |
| `DEBUG` | Enable debug mode | `false` |

---

## 📁 Project Structure

```
iris-model/
├── 📄 README.md              # This file
├── 📄 requirements.txt       # Python dependencies
├── 📄 train.py              # Model training script
├── 📄 app.py                # Streamlit application
├── 📄 docker-compose.yml    # Docker orchestration
├── 📄 config.yaml           # Configuration file
├── 📁 data/                 # Dataset
│   └── iris.csv
├── 📁 models/               # Trained models
│   └── iris_svm.joblib
├── 📁 reports/              # Evaluation reports
│   └── evaluation.json
└── 📁 venv/                 # Virtual environment
```

---

## 🤝 Contributing

<div align="center">

**We welcome contributions! Here's how to help:**

1. 🍴 **Fork** the repository
2. 🌿 **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. ✅ **Test** your changes (`pytest`)
4. 📤 **Push** and 🚀 **Submit** a Pull Request

</div>

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) and [Contribution Guidelines](CONTRIBUTING.md).

---

## 📜 License

<p align="center">
  <img src="https://img.shields.io/github/license/your-org/iris-model?style=for-the-badge&color=green" alt="License">
</p>

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

<div align="center">

| Project | Purpose |
|---------|---------|
| ![Plotly](https://img.shields.io/badge/Plotly-556195?style=for-the-badge&logo=plotly&logoColor=white) | Interactive visualizations |
| ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white) | Web app framework |
| ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white) | Machine learning |
| ![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white) | Containerization |
| **UCI ML Repository** | Iris dataset source |

</div>

---

## 📬 Support

<div align="center">

**Found a bug or have a feature request?**  
Open an [Issue](https://github.com/your-org/iris-model/issues) on GitHub

**Questions?**  
Check the [Discussions](https://github.com/your-org/iris-model/discussions) tab

</div>

---

<div align="center">

# ⭐️ Show some love!

**If this project helped you, please give it a ⭐️ Star!**

![GitHub stars](https://img.shields.io/github/stars/your-org/iris-model?style=social&label=Stars)
![GitHub forks](https://img.shields.io/github/forks/your-org/iris-model?style=social&label=Forks)

---

*Made with ❤️ using Python, Streamlit, and Plotly*

</div>