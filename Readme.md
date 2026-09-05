# 🚗 CarPrice ANN — Used Car Price Predictor

> End-to-end MLOps pipeline for used car price prediction using a Neural Network regression model. Built with production-grade modular architecture — CSV ingestion, KS drift detection, ColumnTransformer preprocessing, ANN trained with EarlyStopping, MLflow experiment tracking via DagsHub, and a FastAPI backend with batch CSV and manual prediction routes.

---

## 🚀 Links

| Resource | URL |
|---|---|
| **Live Demo** | [live url here] |
| **DockerHub** | [hub.docker.com/r/alyanktk/car-price-predictor-ann](https://hub.docker.com/r/alyanktk/car-price-predictor-ann) |
| **MLflow / DagsHub** | [dagshub.com/Alyan-khattak/car-price-predictor-ann](https://dagshub.com/Alyan-khattak/car-price-predictor-ann) |
| **Hugging Face** | [huggingface.co/alyan-ktk/car-price-predictor-ann](https://huggingface.co/alyan-ktk/car-price-predictor-ann) |
| **GitHub** | [github.com/Alyan-khattak/car-price-predictor-ann](https://github.com/Alyan-khattak/car-price-predictor-ann) |

---

## 📊 Model Results

| Metric | Train | Test |
|---|---|---|
| **R² Score** | 0.9295 | 0.9308 |
| **MAE (log scale)** | 0.1381 | 0.1388 |
| **RMSE (log scale)** | 0.1807 | 0.1827 |
| **Overfitting Gap** | — | 0.0013 ✅ |

**Best Architecture:** 44 → 128 → 128 → 1 (linear output, regression)
**Dataset:** 15,354 used car listings from CardDekho · 10 features · 44 after OHE
**Target:** log1p(selling_price) → expm1() at prediction time → actual rupees

> Test R² (0.9308) > Train R² (0.9295) — no overfitting. Dropout + BatchNormalization worked perfectly.

---

## 📌 What This Project Does

Used car pricing is highly variable — the same model can differ by lakhs based on age, km, fuel type, and seller. This pipeline takes 10 key features from a car listing and predicts the fair market selling price using a Neural Network trained on 15k+ real CardDekho listings.

**What makes this an MLOps project, not just a model:**

- **Timestamped artifact versioning** — every run creates `Artifacts/timestamp/` preserving full pipeline history
- **DataValidation as a quality gate** — schema check + Kolmogorov-Smirnov drift test before any model sees data
- **ColumnTransformer** — StandardScaler on 6 numerical + OneHotEncoder on 4 categorical features in one pipeline object
- **EarlyStopping** — training stops when val_loss stops improving, restoring best weights automatically
- **MLflow + DagsHub** — every run logs MAE, RMSE, R² for both train and test; model versioned on HuggingFace
- **Zero cold-start deployment** — model pulled from HuggingFace Hub on startup; never committed to git
- **Modular constants** — no hardcoded values anywhere; everything in `constants/training_pipeline/__init__.py`

---

## 📦 Pipeline

```
Network_Data/cardekho_dataset.csv
    ↓ DataIngestion    → EDA cleaning → log1p transforms → 80/20 split
    ↓ DataValidation   → schema check (11 cols) → KS drift detection → report.yaml
    ↓ DataTransformation → ColumnTransformer (StandardScaler + OHE) → .npy arrays
    ↓ ModelTrainer     → ANN (128→128→1) → EarlyStopping → MLflow → HuggingFace
    ↓ FastAPI          → /predict (CSV batch) + /predict/manual (form)
```

### EDA Cleaning Decisions

| Decision | Reason |
|---|---|
| Drop `Unnamed:0`, `car_name`, `model` | Index/redundant columns |
| Drop `Electric`, `LPG` fuel types | Too few rows, different feature profile |
| Merge `Trustmark Dealer` → `Dealer` | Same category, reduces cardinality |
| Drop `seats=0`, `seats=2` | Data errors / too rare |
| Cap `km_driven` at Q3+1.5×IQR | Remove extreme outliers |
| `log1p(km_driven)` + `log1p(selling_price)` | Reduce skew → better ANN convergence |

### ANN Architecture

```
Input (44 features)
    ↓ Dense(128, relu)
    ↓ BatchNormalization
    ↓ Dropout(0.1)
    ↓ Dense(128, relu)
    ↓ BatchNormalization
    ↓ Dropout(0.1)
    ↓ Dense(1)          ← linear output, no activation (regression)

Loss: MAE · Optimizer: Adam(lr=0.001)
EarlyStopping: patience=10, monitor=val_loss, restore_best_weights=True
```

---

## 🛠️ Setup & Run

### Prerequisites
- Python 3.10
- conda / virtualenv

### 1. Clone

```bash
git clone https://github.com/Alyan-khattak/car-price-predictor-ann.git
cd car-price-predictor-ann
```

### 2. Environment

```bash

python -m venv venv
source venv/bin/activate

    Or ( if using Conda for env)

conda create -n carprice-env python=3.10 -y
conda activate carprice-env

pip install -r requirements.txt
pip install -e .
```

### 3. Environment Variables

Create `.env` in project root:

```env
MLFLOW_TRACKING_USERNAME="your_dagshub_username"
MLFLOW_TRACKING_PASSWORD="your_dagshub_token"
```

### 4. Place Dataset

```bash
mkdir -p Network_Data
# place cardekho_dataset.csv inside Network_Data/
```

### 5. Run Training Pipeline

```bash
PYTHONPATH=. python main.py
```

Artifacts saved to `Artifacts/timestamp/` · Model pushed to HuggingFace automatically.

### 6. Start API Server

```bash
python app.py
# → http://localhost:8000
```

---

## 🐳 Docker

```bash
# Pull and run
docker pull alyanktk/car-price-predictor-ann
docker run -p 8000:8000 alyanktk/car-price-predictor-ann

# Build locally
docker build -t car-price-predictor-ann .
docker run -p 8000:8000 car-price-predictor-ann
```

> On startup, the container automatically pulls `model.keras` and `preprocessing.pkl` from HuggingFace Hub — no model file needed in the repo.

---

## 📡 API Routes

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Landing page |
| `GET` | `/train` | Trigger full training pipeline |
| `GET` | `/predict` | CSV upload form |
| `POST` | `/predict` | Batch CSV → predicted prices table |
| `GET` | `/predict/manual` | Manual 10-field input form |
| `POST` | `/predict/manual` | Single car prediction (JSON) |
| `GET` | `/docs` | Swagger UI |

### Manual Prediction Example

```bash
curl -X POST http://localhost:8000/predict/manual \
  -H "Content-Type: application/json" \
  -d '{
    "brand": "Maruti",
    "vehicle_age": 5,
    "km_driven": 45000,
    "seller_type": "Individual",
    "fuel_type": "Petrol",
    "transmission_type": "Manual",
    "mileage": 18.5,
    "engine": 1197,
    "max_power": 82.0,
    "seats": 5
  }'
```

```json
{
  "predicted_price": 485000,
  "formatted": "₹4,85,000"
}
```

> **Note:** `km_driven` accepts raw kilometres. `log1p()` is applied internally — same transformation used during training.

---

## 📁 Project Structure

```
car-price-predictor-ann/
│
├── app.py                              ← FastAPI backend (6 routes + lifespan)
├── main.py                             ← Pipeline entry point
├── setup.py
├── requirements.txt
├── requirements-prod.txt
├── Dockerfile
├── .env                                ← not committed
│
├── .github/workflows/
│   └── main.yml                        ← GitHub Actions CI
│
├── carprice/
│   ├── components/
│   │   ├── data_ingestion.py           ← CSV → EDA cleaning → split
│   │   ├── data_validation.py          ← schema + KS drift
│   │   ├── data_transformation.py      ← ColumnTransformer → .npy
│   │   └── model_trainer.py            ← ANN train → MLflow → HF push
│   ├── pipeline/
│   │   └── training_pipeline.py        ← TrainingPipeline.run_pipeline()
│   ├── entity/
│   │   ├── config_entity.py            ← 5 config dataclasses
│   │   └── artifact_entity.py          ← 5 artifact dataclasses
│   ├── constants/
│   │   └── training_pipeline/__init__.py  ← all constants
│   ├── utils/
│   │   ├── main_utils/utils.py         ← save/load/yaml/evaluate_ann
│   │   └── dl_utils/
│   │       ├── model/estimator.py      ← CarPriceModel wrapper
│   │       └── metric/regression_metric.py
│   ├── cloud/
│   │   └── hf_syncer.py                ← HuggingFace push/pull
│   ├── exception/
│   └── logging/
│
├── data_schema/
│   └── schema.yaml                     ← 11 expected columns
│
├── templates/
│   ├── index.html                      ← warm premium landing page
│   ├── predict.html                    ← CSV upload form
│   ├── table.html                      ← batch results with ₹ formatting
│   └── predict_manual.html             ← 4-section manual form
│
├── Network_Data/
│   └── cardekho_dataset.csv            ← source dataset
├── Artifacts/                          ← timestamped runs (gitignored)
└── final_model/                        ← model.keras + preprocessing.pkl (gitignored)
```

---

## ⚙️ Key Engineering Decisions

| Decision | Reason |
|---|---|
| `log1p(selling_price)` as target | Reduces skew; ANN trains better on normalized distributions |
| `expm1()` at prediction time | Reverses log transform → actual rupee output |
| `log1p(km_driven)` in ingestion | Raw km_driven is heavily right-skewed |
| `fit_transform` on train only | Prevents data leakage — test stats never seen during fitting |
| `BatchNormalization` | Stabilizes ANN training; normalizes activations per batch |
| `EarlyStopping(restore_best_weights=True)` | Avoids last-epoch model; picks best validation checkpoint |
| `CarPriceModel` wrapper | Preprocessor + model path in one dill object; single load in FastAPI |
| `HuggingFace Hub` model registry | Model never committed to git; pulled on deployment startup |
| `dagshub.init()` inside `track_mlflow()` | Prevents auth prompt on import; only runs when training |

---


## 🧰 Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.10 |
| **Deep Learning** | TensorFlow / Keras · EarlyStopping · BatchNormalization |
| **ML** | scikit-learn · ColumnTransformer · StandardScaler · OneHotEncoder |
| **Tracking** | MLflow · DagsHub |
| **Model Registry** | Hugging Face Hub |
| **API** | FastAPI · uvicorn · Jinja2 · pydantic |
| **Serialization** | dill (preprocessor.pkl) · Keras native (.keras format) |
| **Containerization** | Docker |
| **CI/CD** | GitHub Actions |

---

## 👤 Author

**M. Alyan Khattak**
[github.com/Alyan-khattak](https://github.com/Alyan-khattak) · [portfolio-alyan.vercel.app](https://portfolio-alyan.vercel.app)