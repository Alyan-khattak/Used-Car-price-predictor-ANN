# ═══════════════════════════════════════════════════════════════════
# main.py
# ═══════════════════════════════════════════════════════════════════
# Entry point — pipeline manually test karne ke liye
# Abhi sirf DataIngestion test kar rahe hain
###==============================================================

import sys
from carprice.components.data_ingestion import DataIngestion
from carprice.components.data_validation import DataValidation
from carprice.components.data_transformation import DataTransformation
from carprice.exception.exception import CarPriceException
from carprice.logging.logger import logging
from carprice.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig
)



if __name__ == "__main__":
    try:
        logging.info("Training Pipeline started")

        # STEP 1: Master config
        training_pipeline_config = TrainingPipelineConfig()
        logging.info(f"Artifact dir: {training_pipeline_config.artifact_dir}")

        # STEP 2: DataIngestion
        data_ingestion_config = DataIngestionConfig(
            training_pipeline_config=training_pipeline_config
        )
        data_ingestion = DataIngestion(
            data_ingestion_config=data_ingestion_config
        )
        logging.info("DataIngestion initialized")

        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info(f"DataIngestion completed: {data_ingestion_artifact}")


        # DataValidation
        data_validation_config   = DataValidationConfig(training_pipeline_config)
        data_validation          = DataValidation(
            data_ingestion_artifact=data_ingestion_artifact,
            data_validation_config=data_validation_config
        )
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info(f"DataValidation: {data_validation_artifact}")


        # DataTransformation
        data_transformation_config   = DataTransformationConfig(training_pipeline_config)
        data_transformation          = DataTransformation(
            data_validation_artifact=data_validation_artifact,
            data_transformation_config=data_transformation_config
        )
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info(f"DataTransformation: {data_transformation_artifact}")

        print(data_transformation_artifact)

        
    except Exception as e:
        raise CarPriceException(e, sys)