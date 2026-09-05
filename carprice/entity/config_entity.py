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




# ══════════════════════════════════════════════════════════════════
# CLASS 3: DataValidationConfig
# ══════════════════════════════════════════════════════════════════
class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """
        PATH STRUCTURE:
        Artifacts/timestamp/
        └── data_validation/
            ├── validated/train.csv + test.csv  ← DataTransformation ka INPUT
            ├── invalid/train.csv + test.csv
            └── drift_report/report.yaml
        """
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_VALIDATON_DIR_NAME
        )
        # → "Artifacts/timestamp/data_validation": 

        self.valid_data_dir: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_VALID_DIR
        )
        # → "Artifacts/timestamp/data_validation/validated"

        self.invalid_data_dir: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_INVALID_DIR
        )
        # → "Artifacts/timestamp/data_validation/invalid"

        self.valid_train_file_path: str = os.path.join(
            self.valid_data_dir,
            training_pipeline.TRAIN_FILE_NAME
        )
        # → "Artifacts/timestamp/data_validation/validated/train.csv"

        self.valid_test_file_path: str = os.path.join(
            self.valid_data_dir,
            training_pipeline.TEST_FILE_NAME
        )
        # → "Artifacts/timestamp/data_validation/validated/test.csv"

        self.invalid_train_file_path: str = os.path.join(
            self.invalid_data_dir,
            training_pipeline.TRAIN_FILE_NAME
        )
        # → "Artifacts/timestamp/data_validation/invalid/train.csv"

        self.invalid_test_file_path: str = os.path.join(
            self.invalid_data_dir,
            training_pipeline.TEST_FILE_NAME
        )
        # → "Artifacts/timestamp/data_validation/invalid/test.csv"

        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME
        )
        # → "Artifacts/timestamp/data_validation/drift_report/report.yaml"




# ══════════════════════════════════════════════════════════════════
# CLASS 4: DataTransformationConfig
# ══════════════════════════════════════════════════════════════════
class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """
        PATH STRUCTURE:
        Artifacts/timestamp/
        └── data_transformation/
            ├── transformed/
            │   ├── train.npy  ← ModelTrainer ka INPUT (N_train × 45)
            │   └── test.npy   ← ModelTrainer ka INPUT (N_test × 45)
            └── transformed_object/
                └── preprocessing.pkl  ← ColumnTransformer (fitted)
        """
        self.data_transformation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_TRANSFORMATION_DIR_NAME
        )
        # → "Artifacts/timestamp/data_transformation"

        self.transformed_train_file_path: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TRAIN_FILE_NAME.replace("csv", "npy")
        )
        # → "Artifacts/timestamp/data_transformation/transformed/train.npy"
        # .csv → .npy — numpy binary format for ANN input

        self.transformed_test_file_path: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TEST_FILE_NAME.replace("csv", "npy")
        )
        # → "Artifacts/timestamp/data_transformation/transformed/test.npy"

        self.transformed_object_file_path: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            training_pipeline.PREPROCESSING_OBJECT_FILE_NAME
        )
        # → "Artifacts/timestamp/data_transformation/transformed_object/preprocessing.pkl"
        # fitted ColumnTransformer → PredictPipeline load karega

