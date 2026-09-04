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
