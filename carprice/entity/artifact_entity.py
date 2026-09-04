# ═══════════════════════════════════════════════════════════════════
# carprice/entity/config_entity.py
# ═══════════════════════════════════════════════════════════════════
# Har pipeline component ka CONFIG yahan define hota hai
# Constants se raw values leke paths banata hai
#
# PATTERN:
# constants/ → raw values (strings, floats)
#      ↓
# config_entity.py → os.path.join se paths banao
#      ↓
# component → config object lo, kaam karo
###==============================================================

from datetime import datetime
import os
from carprice.constants import training_pipeline


# ══════════════════════════════════════════════════════════════════
# CLASS 1: TrainingPipelineConfig
# ══════════════════════════════════════════════════════════════════
# Master config — timestamp generate karta hai
# Sab doosre configs isko inject karte hain
# Taaki sab ek hi Artifacts/timestamp/ folder mein jaayein


class TrainingPipelineConfig:
    def __init__(self, timestamp=datetime.now()):
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")

        self.pipeline_name: str = training_pipeline.PIPELINE_NAME
        # → "CarPricePredictor"

        self.artifact_name: str = training_pipeline.ARTIFACT_DIR
        # → "Artifacts"

        self.artifact_dir: str = os.path.join(
            self.artifact_name, timestamp
        )
        # → "Artifacts/08_24_2026_14_32_00"
        # IMP: har run pe NAYA folder → history preserve

        self.timestamp: str = timestamp




# ══════════════════════════════════════════════════════════════════
# CLASS 2: DataIngestionConfig
# ══════════════════════════════════════════════════════════════════
# MongoDB nahi — CSV se directly padhenge
# Network_Data/cardekho_dataset.csv → feature_store → train/test split

class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """
        PATH STRUCTURE:
        Artifacts/timestamp/
        └── data_ingestion/
            ├── feature_store/cardekho_dataset.csv  ← raw CSV copy
            └── ingested/
                ├── train.csv  ← DataValidation ka INPUT
                └── test.csv
        """
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_INGESTION_DIR_NAME
        )
        # → "Artifacts/timestamp/data_ingestion"

        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,
            training_pipeline.FILE_NAME
        )
        # → "Artifacts/timestamp/data_ingestion/feature_store/cardekho_dataset.csv"

        self.training_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TRAIN_FILE_NAME
        )
        # → "Artifacts/timestamp/data_ingestion/ingested/train.csv"

        self.testing_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TEST_FILE_NAME
        )
        # → "Artifacts/timestamp/data_ingestion/ingested/test.csv"

        self.train_test_split_ratio: float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        # → 0.2

        self.random_state: int = training_pipeline.DATA_INGESTION_RANDOM_STATE
        # → 42
