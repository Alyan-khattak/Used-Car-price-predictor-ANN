# ═══════════════════════════════════════════════════════════════════
# carprice/entity/artifact_entity.py
# ═══════════════════════════════════════════════════════════════════
# Har pipeline component ka RETURN TYPE yahan define hota hai
# @dataclass → typed objects — never plain tuples
#
# CHAIN:
# DataIngestionArtifact
#   → DataValidationArtifact
#     → DataTransformationArtifact
#       → ModelTrainerArtifact
###==============================================================

from dataclasses import dataclass

# ── ARTIFACT 1: DataIngestionArtifact ────────────────────────────
# DataIngestion.initiate_data_ingestion() return karta hai
# DataValidation ko milega@dataclass
@dataclass
class DataIngestionArtifact:
    train_file_path: str
    # "Artifacts/timestamp/data_ingestion/ingested/train.csv"

    test_file_path: str
    # "Artifacts/timestamp/data_ingestion/ingested/test.csv"




# ── ARTIFACT 2: DataValidationArtifact ───────────────────────────
# DataValidation.initiate_data_validation() return karta hai
# DataTransformation ko milega
@dataclass
class DataValidationArtifact:
    validation_status: bool
    # True = valid → pipeline continue
    # False = invalid → pipeline flag

    valid_train_file_path: str
    # "Artifacts/timestamp/data_validation/validated/train.csv"

    valid_test_file_path: str
    # "Artifacts/timestamp/data_validation/validated/test.csv"

    invalid_train_file_path: str
    # "Artifacts/timestamp/data_validation/invalid/train.csv"

    invalid_test_file_path: str
    # "Artifacts/timestamp/data_validation/invalid/test.csv"

    drift_report_file_path: str
    # "Artifacts/timestamp/data_validation/drift_report/report.yaml"




# ── ARTIFACT 3: DataTransformationArtifact ───────────────────────
# DataTransformation.initiate_data_transformation() return karta hai
# ModelTrainer ko milega
@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    # "Artifacts/timestamp/data_transformation/transformed_object/preprocessing.pkl"
    # fitted ColumnTransformer (StandardScaler + OHE)
    # PredictPipeline load karega

    transformed_train_file_path: str
    # "Artifacts/timestamp/data_transformation/transformed/train.npy"
    # shape: (N_train, 45) → 44 features + 1 target (last col)

    transformed_test_file_path: str
    # "Artifacts/timestamp/data_transformation/transformed/test.npy"
    # shape: (N_test, 45)