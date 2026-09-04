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

# ── ARTIFACT 1: DataIngestionArtifact ────────────────────────────
# DataIngestion.initiate_data_ingestion() return karta hai
# DataValidation ko milega@dataclass
class DataIngestionArtifact:
    train_file_path: str
    # "Artifacts/timestamp/data_ingestion/ingested/train.csv"

    test_file_path: str
    # "Artifacts/timestamp/data_ingestion/ingested/test.csv"