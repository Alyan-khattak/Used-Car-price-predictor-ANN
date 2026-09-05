# ═══════════════════════════════════════════════════════════════════
# carprice/constants/training_pipeline/__init__.py
# ═══════════════════════════════════════════════════════════════════
# SAB hardcoded values yahan hain — koi bhi component mein nahi
# Koi bhi value change karni ho → sirf yeh file change karo
# import training_pipeline as tp → tp.CONSTANT_NAME
###==============================================================

import os
import numpy as np


# ─────────────────────────────────────────────────────────────────
# COMMON
# ─────────────────────────────────────────────────────────────────
TARGET_COLUMN:   str = "selling_price"
# log1p transformed target — expm1() at prediction time
# IMP: model log scale pe train hua hai
# predict karte waqt expm1(prediction) → actual rupees

PIPELINE_NAME:   str = "CarPricePredictor"
ARTIFACT_DIR:    str = "Artifacts"
FILE_NAME:       str = "cardekho_dataset.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME:  str = "test.csv"
SCHEMA_FILE_PATH: str = os.path.join("data_schema", "schema.yaml")


# ─────────────────────────────────────────────────────────────────
# DATA INGESTION
# MongoDB nahi hai — CSV se directly padhenge
# ─────────────────────────────────────────────────────────────────

DATA_INGESTION_DIR_NAME:str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR:str = "feature_store"
DATA_INGESTION_INGESTED_DIR:str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO:float = 0.2
DATA_INGESTION_RANDOM_STATE:int = 42


# EDA decisions → cleaning constants
# Yeh values DataIngestion cleaning mein use hongi
FUEL_TYPES_TO_DROP:   list = ["Electric", "LPG"]
SEATS_TO_DROP:        list = [0, 2]
SELLER_TYPE_MERGE:    dict = {"Trustmark Dealer": "Dealer"}
COLUMNS_TO_DROP:      list = ["Unnamed: 0", "car_name", "model"]
KM_DRIVEN_CAP_QUANTILE: float = 0.75   # Q3 + 1.5*IQR cap



# ─────────────────────────────────────────────────────────────────
# DATA VALIDATION
# ─────────────────────────────────────────────────────────────────
DATA_VALIDATON_DIR_NAME:                str = "data_validation"
DATA_VALIDATION_VALID_DIR:              str = "validated"
DATA_VALIDATION_INVALID_DIR:            str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR:       str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"




# ─────────────────────────────────────────────────────────────────
# DATA TRANSFORMATION
# ColumnTransformer: StandardScaler + OHE
# ─────────────────────────────────────────────────────────────────
DATA_TRANSFORMATION_DIR_NAME:               str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR:   str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"
PREPROCESSING_OBJECT_FILE_NAME:             str = "preprocessing.pkl"

# Column names — schema se alag yahan bhi define karo
# DataTransformation mein ColumnTransformer ko chahiye
NUMERICAL_COLUMNS: list = [
    "vehicle_age", "km_driven", "mileage",
    "engine", "max_power", "seats"
]

CATEGORICAL_COLUMNS: list = [
    "brand", "seller_type",
    "fuel_type", "transmission_type"
]







# ─────────────────────────────────────────────────────────────────
# ANN MODEL TRAINER
# TensorFlow/Keras — no GridSearchCV
# ─────────────────────────────────────────────────────────────────
MODEL_TRAINER_DIR_NAME:       str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.keras"
# IMP: .keras format — tensorflow recommend karta hai .h5 se better
# SavedModel format bhi use kar sakte hain

PREPROCESSING_FILE_NAME: str = "preprocessing.pkl"


ANN_DROPOUT_RATE:    float = 0.1
ANN_LEARNING_RATE:   float = 0.001
ANN_BATCH_SIZE:      int   = 32
ANN_LOSS:            str   = "mae"
# MAE loss → regression → car price mein MAE interpretable hai (rupees mein)

ANN_OPTIMIZER:       str   = "adam"

# EarlyStopping
ANN_EARLY_STOPPING_PATIENCE:       int  = 10
ANN_EARLY_STOPPING_MONITOR:        str  = "val_loss"
ANN_EARLY_STOPPING_RESTORE_BEST:   bool = True
# IMP: restore_best_weights=True → best epoch ka model milega
# patience=10 → 10 epochs mein improvement nahi → stop

# Model quality thresholds
MODEL_TRAINER_EXPECTED_R2_SCORE:   float = 0.80
# agar R² < 0.80 → model reject karo
MODEL_TRAINER_OVERFITTING_THRESHOLD: float = 0.05
# |train_r2 - test_r2| > 0.05 → overfitting warning

# ─────────────────────────────────────────────────────────────────
# HUGGING FACE
# ─────────────────────────────────────────────────────────────────
HF_REPO_ID:   str = "alyan-ktk/car-price-predictor-ann"
HF_REPO_TYPE: str = "model"
HF_MODEL_DIR: str = "final_model/"