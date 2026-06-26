---
title: ML Platform
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# ML Platform

**Kaggle-style Machine Learning Platform** — Upload datasets, clean & analyze data, train ML models, evaluate them, and make predictions — all from a sleek HUD/Cyber-themed dashboard.

Built with **React 18 + Vite 6** on the frontend and **FastAPI + scikit-learn** on the backend.

---

## 🌟 Features at a Glance

| Category | Features |
|----------|----------|
| 📂 **Data** | Upload CSV/Excel/JSON/Parquet, load sample datasets (Iris, Titanic, Boston Housing, etc.), preview table, column types, missing value detection |
| 🧹 **Cleaning** | Drop/fill missing values, label/one-hot encoding, normalize, standardize, log transform, cap outliers |
| 📊 **Analysis** | Descriptive statistics, correlation matrix, outlier detection, skewness, histograms |
| 🤖 **Training** | 9+ classification & regression algorithms, AutoML, configurable test size, async training with progress bar |
| 🔘 **Clustering** | K-Means (with elbow plot), DBSCAN, Agglomerative Clustering |
| 📉 **Dim Reduction** | PCA, t-SNE, SVD with scatter plot visualization |
| 🧠 **Neural Networks** | MLPClassifier/MLPRegressor with configurable hidden layers |
| ⚙️ **HP Tuning** | GridSearchCV for Random Forest, SVM, Logistic Regression, Gradient Boosting |
| ♻️ **K-Fold CV** | Cross-validation with per-fold bar chart visualization |
| 🔧 **Feature Engineering** | Polynomial features, interaction terms, binning, log transform, rolling mean |
| 📋 **Evaluation** | Accuracy, precision, recall, F1, AUC-ROC, RMSE, MAE, R², MAPE, confusion matrix, ROC curve, feature importance |
| 🔮 **Prediction** | Single prediction with class probabilities and confidence score |
| 📦 **Model Management** | List all models, view metadata, delete, export (.joblib) |
| 📈 **Visualization** | Histograms, scatter plots, correlation heatmaps, box plots, bar charts (all custom SVG) |

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 18, Vite 6, Zustand (state), Framer Motion (animations), react-data-table-component, react-dropzone |
| **Backend** | FastAPI, Uvicorn, Python 3.11+ |
| **ML Libraries** | scikit-learn, XGBoost, pandas, numpy, joblib, scipy |
| **Visualization** | matplotlib, seaborn (backend), custom SVG (frontend) |
| **Infrastructure** | Docker, docker-compose, python-dotenv |
| **Theme** | Custom HUD/Cyber theme — Orbitron font, animated grid, scanline effect, neon accents |

---

## 📁 Project Structure

```
ml_platform/
│
├── run.py                            # Entry point — auto-install deps + start server
├── .env.example                      # Environment configuration template
├── .gitignore
├── Dockerfile                        # Docker image
├── docker-compose.yml                # Docker orchestration
├── README.md                         # This file
│
├── backend/                          # Python FastAPI backend
│   ├── main.py                       # API routes, CORS, middleware, job system
│   ├── config.py                     # Centralized configuration (env vars)
│   ├── auth.py                       # API key authentication middleware
│   ├── requirements.txt              # Python dependencies
│   ├── uploads/                      # Uploaded datasets directory
│   └── agents/ml/
│       ├── __init__.py               # Package exports
│       ├── data_engine.py            # Data loading, cleaning, analysis, caching
│       ├── model_factory.py          # Model training, clustering, tuning, feature engineering
│       ├── evaluator.py              # Evaluation, prediction, model management
│       └── visualizer.py             # Charts, feature importance, confusion matrix, ROC
│
└── frontend-react/                   # React + Vite frontend
    ├── package.json
    ├── vite.config.js
    ├── index.html
    ├── public/
    ├── dist/                         # Production build
    └── src/
        ├── main.jsx                  # React mount point
        ├── App.jsx                   # Root component (HUD layout, particles, clock)
        ├── index.css                 # Global CSS (HUD/Cyber theme)
        ├── components/
        │   ├── MLPanel.jsx           # Main app — 13 tabs (Data, Train, Evaluate, etc.)
        │   └── HUDComponents.jsx     # Shared UI (Button, Card, Modal, ErrorBlock, etc.)
        ├── hooks/
        │   └── useJobPolling.js      # Shared hook for async job polling
        ├── services/
        │   └── api.js                # API client with timeout & error handling
        └── store/
            └── useAtlasStore.js       # Zustand store with persist middleware
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend development)

### One-Command Start

```bash
python run.py
```

This installs Python dependencies and starts the server at **http://localhost:8000**.

### Manual Setup

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend (optional — for development)
cd frontend-react
npm install
npm run dev
```

### Docker

```bash
docker-compose up --build
```

### Environment Configuration

```bash
cp .env.example .env
# Edit .env as needed
```

---

## 📖 How to Use (Step by Step)

### Train a Model — Complete Workflow

**Step 1: Upload Data**
- Go to the **Data** tab
- Drag & drop a CSV/Excel/JSON file, or click "Load Sample Dataset" (Iris, Titanic, etc.)
- Preview the data and check column types

**Step 2: Clean Data (if needed)**
- In the **Data** tab, use cleaning options
- Drop missing rows, fill missing values, encode categorical columns
- Normalize or standardize numerical columns
- Click "Apply Cleaning"

**Step 3: Analyze Data (optional)**
- Go to the **Analyze** tab
- Click "Analyze Dataset"
- View descriptive statistics, correlations, outliers, skewness

**Step 4: Train a Model**
- Go to the **Train** tab
- Select **Target Column** (what you want to predict)
- Select **Task Type**: Classification / Regression / AutoML
- Set **Test Size** (default: 20%)
- Select **Algorithm** or choose "Auto" for best match
- Click "Train Model" and watch the progress bar

**Step 5: Evaluate**
- Go to the **Evaluate** tab
- Select the trained model
- View accuracy, precision, recall, F1 (classification) or RMSE, R² (regression)
- View confusion matrix, ROC curve, feature importance

**Step 6: Predict**
- Go to the **Predict** tab
- Select the trained model
- Enter feature values
- Click "Predict" to see the result with confidence/probabilities

---

## 📚 Supported Algorithms

### Classification (9+)
Random Forest, Gradient Boosting, Logistic Regression, SVM, Naive Bayes, KNN, Decision Tree, Extra Trees, AdaBoost, XGBoost

### Regression (9+)
Random Forest, Gradient Boosting, Linear Regression, Ridge, Lasso, Elastic Net, SVR, KNN, Extra Trees, XGBoost

### Clustering
K-Means, DBSCAN, Agglomerative

### Dimensionality Reduction
PCA, t-SNE, TruncatedSVD

### Neural Networks
MLPClassifier, MLPRegressor

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ml/upload` | Upload dataset file |
| POST | `/ml/sample` | Load sample dataset |
| POST | `/ml/clean` | Apply cleaning operations |
| POST | `/ml/analyze` | Compute statistics |
| POST | `/ml/train` | Start model training |
| GET | `/ml/job/{id}` | Poll job status |
| POST | `/ml/evaluate` | Evaluate a model |
| POST | `/ml/predict` | Make a prediction |
| GET | `/ml/models` | List all models |
| POST | `/ml/delete-model` | Delete a model |
| POST | `/ml/cluster` | Run clustering |
| POST | `/ml/dim-reduce` | Dimensionality reduction |
| POST | `/ml/neural` | Train neural network |
| POST | `/ml/tune` | Hyperparameter tuning |
| POST | `/ml/kfold` | K-Fold cross-validation |
| POST | `/ml/feature-engineer` | Feature engineering |
| POST | `/ml/chart-data` | Chart data |
| POST | `/ml/confusion-matrix` | Confusion matrix |
| POST | `/ml/roc-curve` | ROC curve |
| POST | `/ml/elbow-plot` | Elbow plot |
| POST | `/ml/feature-importance` | Feature importance |
| POST | `/ml/export-model` | Export model |
| GET | `/health` | Health check |

---

## 🌍 Deployment (Live — HuggingFace Spaces)

**[HuggingFace Spaces](https://huggingface.co/spaces)** — সম্পূর্ণ ফ্রি, Docker support, ML-optimized, **কখনো sleep হয় না**।

### এক ক্লিকে Deploy করুন:

### Step 1: Push to GitHub

```bash
git push origin main
```

### Step 2: Create HuggingFace Space

1. **[huggingface.co](https://huggingface.co)** → Sign up (GitHub account দিয়ে)
2. উপরের ডান দিকে আপনার profile pic → **New Space**
3. **Space Name:** `ml-platform` (যেকোনো নাম দিন)
4. **License:** `MIT`
5. **Space SDK:** **Docker** সিলেক্ট করুন
6. **Docker Template:** **Blank** সিলেক্ট করুন
7. **Create Space** বাটনে ক্লিক করুন

### Step 3: Connect GitHub Repository

1. Space created হওয়ার পর **Settings** tab এ যান
2. **Repository link** section → **Connect Git repository**
3. আপনার GitHub account authorize করুন
4. আপনার `ml-platform` repo সিলেক্ট করুন
5. **Save** দিন

### Step 4: Auto-Deploy

HuggingFace auto-detect করবে `Dockerfile` এবং build শুরু করবে।
- **Build time:** ~5-8 মিনিট (প্রথমবার)
- **Live URL:** `https://your-username-ml-platform.hf.space`
- ✅ **Auto-update:** GitHub-এ push করলেই auto-deploy হবে

> **⏱️ Auto-sleep করে না — 24/7 ফ্রি চলবে!**

### Environment Variables (Optional)

Space এর **Settings → Repository secrets** এ এগুলো set করতে পারেন:

| Variable | Recommended Value | Why |
|----------|------------------|-----|
| `MAX_UPLOAD_SIZE_MB` | `10` | Free tier memory limit |
| `LOG_LEVEL` | `WARNING` | Reduce log noise |
| `MAX_CONCURRENT_JOBS` | `1` | Stay within free tier RAM |
| `API_KEY` | (optional) | Set for API key authentication |

---

## 🔧 Configuration

All settings are configurable via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `DEBUG` | `true` | Enable hot-reload |
| `API_KEY` | (none) | Enable API key auth |
| `MAX_UPLOAD_SIZE_MB` | `100` | Max upload file size |
| `MAX_CONCURRENT_JOBS` | `3` | Max parallel training jobs |
| `CACHE_MAXSIZE` | `10` | Dataset cache size |
| `CACHE_TTL_SECONDS` | `300` | Cache TTL |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## 📝 License

This project is open source. See the `LICENSE` file for details.

---

> **Made with ❤️ for the ML community**
>
> [GitHub Repository](https://github.com/hridoy8098/ml-platform)
