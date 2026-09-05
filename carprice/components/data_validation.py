# ═══════════════════════════════════════════════════════════════════
# carprice/components/data_validation.py
# ═══════════════════════════════════════════════════════════════════
# DataIngestion ke baad data quality check karta hai.
# Schema validate karta hai aur data drift detect karta hai.
#
# NETWORKSECURITY SE SAME — sirf import paths alag hain
#
# FLOW:
# DataIngestionArtifact (train/test paths)
#       ↓ read_data()
# train_df, test_df
#       ↓ validate_number_cols()
# schema.yaml se expected columns → actual se compare
#       ↓ validate_column_types()
# dtypes check karo
#       ↓ detect_data_drift()
# KS test → train vs test → report.yaml
#       ↓ initiate_data_validation()
# DataValidationArtifact
###==============================================================

import os
import sys
import pandas as pd
from scipy.stats import ks_2samp

from carprice.entity.config_entity import DataValidationConfig
from carprice.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact
)
from carprice.exception.exception import CarPriceException
from carprice.logging.logger import logging
from carprice.constants.training_pipeline import SCHEMA_FILE_PATH
from carprice.utils.main_utils.utils import (
    read_yaml_file,
    write_yaml_file
)


class DataValidation:
    def __init__(self,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        """
        Parameters:
            data_ingestion_artifact : DataIngestion ka output
                ├── train_file_path
                └── test_file_path

            data_validation_config : config_entity.py se
                ├── valid_train_file_path
                ├── valid_test_file_path
                ├── invalid_train_file_path
                ├── invalid_test_file_path
                └── drift_report_file_path
        """
        try:
            self.data_ingestion_artifact  = data_ingestion_artifact
            self.data_validation_config   = data_validation_config
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            # schema_config → {"columns": [...], "numerical_columns": [...]}
            logging.info(f"Schema loaded from: {SCHEMA_FILE_PATH}")
        except Exception as e:
            raise CarPriceException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """
        CSV file padhta hai → DataFrame return karta hai.

        @staticmethod kyun?
        self use nahi karta — pure utility
        DataValidation.read_data() directly call kar sakte hain
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CarPriceException(e, sys)

    def validate_number_cols(self, dataframe: pd.DataFrame) -> bool:
        """
        DataFrame mein expected columns hain ya nahi.

        schema.yaml mein columns list define hai
        incoming df ke columns se compare karo

        Returns:
            bool: True = columns match ✅ | False = mismatch ❌
        """
        try:
            expected_cols = [
                list(col.keys())[0]
                for col in self.schema_config["columns"]
            ]
            actual_cols = dataframe.columns.tolist()

            missing = set(expected_cols) - set(actual_cols)
            extra   = set(actual_cols) - set(expected_cols)

            logging.info(f"Expected cols : {len(expected_cols)}")
            logging.info(f"Actual cols   : {len(actual_cols)}")

            if missing:
                logging.warning(f"Missing columns: {missing}")
            if extra:
                logging.warning(f"Extra columns: {extra}")

            return len(missing) == 0

        except Exception as e:
            raise CarPriceException(e, sys)

    def validate_column_types(self, dataframe: pd.DataFrame) -> bool:
        """
        Numerical columns exist karte hain ya nahi check karta hai.

        Returns:
            bool: True = sab numerical cols hain ✅
        """
        try:
            expected_num = self.schema_config["numerical_columns"]
            actual_num   = dataframe.select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist()

            logging.info(f"Expected numerical: {expected_num}")
            logging.info(f"Actual numerical  : {actual_num}")

            for col in expected_num:
                if col not in actual_num:
                    logging.warning(f"Missing numerical col: {col}")
                    return False
            return True

        except Exception as e:
            raise CarPriceException(e, sys)

    def detect_data_drift(self,
                          base_df: pd.DataFrame,
                          current_df: pd.DataFrame,
                          threshold: float = 0.05) -> bool:
        """
        KS test se train aur test distribution compare karta hai.

        p_value >= 0.05 → same distribution → no drift ✅
        p_value <  0.05 → different distribution → DRIFT ⚠️

        Parameters:
            base_df    : train DataFrame (reference)
            current_df : test DataFrame  (compare)
            threshold  : p_value threshold (default 0.05)

        Returns:
            bool: True = no drift ✅ | False = drift detected ⚠️
        """
        try:
            status = True
            report = {}

            # sirf numerical columns pe KS test karo
            numerical_cols = self.schema_config["numerical_columns"]

            for column in numerical_cols:
                if column not in base_df.columns:
                    continue

                d1 = base_df[column]
                d2 = current_df[column]

                is_same_dist = ks_2samp(d1, d2)

                if threshold <= is_same_dist.pvalue:
                    is_found = False   # no drift
                else:
                    is_found = True    # drift detected
                    status   = False

                report[column] = {
                    "p_value":      float(is_same_dist.pvalue),
                    "drift_status": is_found
                }

            # drift report YAML mein save karo
            drift_report_file_path = \
                self.data_validation_config.drift_report_file_path
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)

            write_yaml_file(
                file_path=drift_report_file_path,
                content=report
            )
            logging.info(f"Drift report saved: {drift_report_file_path}")
            logging.info(f"Drift status: {'No Drift ✅' if status else 'Drift Detected ⚠️'}")

            return status

        except Exception as e:
            raise CarPriceException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        DataValidation ka main entry point.

        Returns:
            DataValidationArtifact:
                ├── validation_status
                ├── valid_train_file_path
                ├── valid_test_file_path
                ├── invalid_train_file_path
                ├── invalid_test_file_path
                └── drift_report_file_path
        """
        try:
            logging.info("=" * 50)
            logging.info("DataValidation started")
            logging.info("=" * 50)

            error_message = ""

            # ── STEP 1: Data padhna ───────────────────────────────
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path  = self.data_ingestion_artifact.test_file_path

            train_df = DataValidation.read_data(train_file_path)
            test_df  = DataValidation.read_data(test_file_path)
            logging.info(
                f"Train: {train_df.shape} | Test: {test_df.shape}"
            )

            # ── STEP 2: Column count validate ────────────────────
            status = self.validate_number_cols(train_df)
            if not status:
                error_message += "Train: missing columns\n"

            status = self.validate_number_cols(test_df)
            if not status:
                error_message += "Test: missing columns\n"

            # ── STEP 3: Numerical cols validate ──────────────────
            status = self.validate_column_types(train_df)
            if not status:
                error_message += "Train: numerical cols missing\n"

            status = self.validate_column_types(test_df)
            if not status:
                error_message += "Test: numerical cols missing\n"

            # ── STEP 4: Drift detection ───────────────────────────
            drift_status = self.detect_data_drift(
                base_df=train_df,
                current_df=test_df
            )

            # ── STEP 5: Valid data save karo ──────────────────────
            dir_path = os.path.dirname(
                self.data_validation_config.valid_train_file_path
            )
            os.makedirs(dir_path, exist_ok=True)

            train_df.to_csv(
                self.data_validation_config.valid_train_file_path,
                index=False, header=True
            )
            test_df.to_csv(
                self.data_validation_config.valid_test_file_path,
                index=False, header=True
            )

            logging.info(
                f"Valid data saved:\n"
                f"  train → {self.data_validation_config.valid_train_file_path}\n"
                f"  test  → {self.data_validation_config.valid_test_file_path}"
            )

            # ── STEP 6: Artifact banao ────────────────────────────
            data_validation_artifact = DataValidationArtifact(
                validation_status=drift_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            logging.info(f"DataValidation completed: {data_validation_artifact}")
            logging.info("=" * 50)

            return data_validation_artifact

        except Exception as e:
            raise CarPriceException(e, sys)