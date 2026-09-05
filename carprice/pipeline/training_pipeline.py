# ═══════════════════════════════════════════════════════════════════
# carprice/pipeline/training_pipeline.py
# ═══════════════════════════════════════════════════════════════════
# Poori training pipeline ko ek class mein wrap karta hai.
# main.py mein jo manually kiya tha → ab yahan class mein
#
# FastAPI GET /train → TrainingPipeline().run_pipeline() call karega
#
# FLOW:
# run_pipeline()
#   ↓ start_data_ingestion()   → DataIngestionArtifact
#   ↓ start_data_validation()  → DataValidationArtifact
#   ↓ start_data_transformation() → DataTransformationArtifact
#   ↓ start_model_trainer()    → ModelTrainerArtifact
###==============================================================

import sys

from carprice.components.data_ingestion     import DataIngestion
from carprice.components.data_validation    import DataValidation
from carprice.components.data_transformation import DataTransformation
from carprice.components.model_trainer      import ModelTrainer

from carprice.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)

from carprice.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact
)

from carprice.exception.exception import CarPriceException
from carprice.logging.logger import logging


class TrainingPipeline:
    def __init__(self):
        # IMP: ek hi TrainingPipelineConfig
        # sab steps same timestamp use karenge
        # taaki sab Artifacts/timestamp/ ek folder mein jaayein
        self.training_pipeline_config = TrainingPipelineConfig()
        logging.info(
            f"TrainingPipeline initialized | "
            f"artifact_dir: {self.training_pipeline_config.artifact_dir}"
        )

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        CSV padhna → EDA cleaning → feature store → train/test split

        Returns:
            DataIngestionArtifact:
                ├── train_file_path
                └── test_file_path
        """
        try:
            data_ingestion_config = DataIngestionConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            logging.info("DataIngestion started")

            data_ingestion = DataIngestion(
                data_ingestion_config=data_ingestion_config
            )
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            logging.info(
                f"DataIngestion completed: {data_ingestion_artifact}\n"
            )
            return data_ingestion_artifact

        except Exception as e:
            raise CarPriceException(e, sys)

    def start_data_validation(
            self,
            data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        """
        Schema check → KS drift detection → valid data save

        Parameters:
            data_ingestion_artifact : start_data_ingestion() ka output

        Returns:
            DataValidationArtifact
        """
        try:
            data_validation_config = DataValidationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            logging.info("DataValidation started")

            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=data_validation_config
            )
            data_validation_artifact = data_validation.initiate_data_validation()

            logging.info(
                f"DataValidation completed: {data_validation_artifact}\n"
            )
            return data_validation_artifact

        except Exception as e:
            raise CarPriceException(e, sys)

    def start_data_transformation(
            self,
            data_validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:
        """
        ColumnTransformer → numpy arrays → preprocessing.pkl

        Parameters:
            data_validation_artifact : start_data_validation() ka output

        Returns:
            DataTransformationArtifact
        """
        try:
            data_transformation_config = DataTransformationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            logging.info("DataTransformation started")

            data_transformation = DataTransformation(
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=data_transformation_config
            )
            data_transformation_artifact = \
                data_transformation.initiate_data_transformation()

            logging.info(
                f"DataTransformation completed: {data_transformation_artifact}\n"
            )
            return data_transformation_artifact

        except Exception as e:
            raise CarPriceException(e, sys)

    def start_model_trainer(
            self,
            data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        """
        GridSearch → best params → ANN train → MLflow → HF push

        Parameters:
            data_transformation_artifact : start_data_transformation() ka output

        Returns:
            ModelTrainerArtifact
        """
        try:
            model_trainer_config = ModelTrainerConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            logging.info("ModelTrainer started")

            model_trainer = ModelTrainer(
                model_trainer_config=model_trainer_config,
                data_transformation_artifact=data_transformation_artifact
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()

            logging.info(
                f"ModelTrainer completed: {model_trainer_artifact}\n"
            )
            return model_trainer_artifact

        except Exception as e:
            raise CarPriceException(e, sys)

    def run_pipeline(self) -> ModelTrainerArtifact:
        """
        Poori pipeline ek call mein run karta hai.
        FastAPI GET /train yahan se call karega.

        Returns:
            ModelTrainerArtifact
        """
        try:
            logging.info("=" * 60)
            logging.info("Training Pipeline Started")
            logging.info("=" * 60)

            # ── STEP 1 ─────────────────────────────────────────────
            data_ingestion_artifact = self.start_data_ingestion()

            # ── STEP 2 ─────────────────────────────────────────────
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )

            # ── STEP 3 ─────────────────────────────────────────────
            data_transformation_artifact = self.start_data_transformation(
                data_validation_artifact=data_validation_artifact
            )

            # ── STEP 4 ─────────────────────────────────────────────
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )

            logging.info("=" * 60)
            logging.info("Training Pipeline Completed")
            logging.info("=" * 60)

            return model_trainer_artifact

        except Exception as e:
            raise CarPriceException(e, sys)