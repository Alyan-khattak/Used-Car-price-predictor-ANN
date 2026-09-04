# ═══════════════════════════════════════════════════════════════════
# main.py
# ═══════════════════════════════════════════════════════════════════
# Entry point — pipeline manually test karne ke liye
# Abhi sirf DataIngestion test kar rahe hain
###==============================================================

import sys
from carprice.components.data_ingestion import DataIngestion
from carprice.exception.exception import CarPriceException
from carprice.logging.logger import logging
from carprice.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig
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

        print(data_ingestion_artifact)

    except Exception as e:
        raise CarPriceException(e, sys)