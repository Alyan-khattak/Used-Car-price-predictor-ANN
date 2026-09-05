# ═══════════════════════════════════════════════════════════════════
# carprice/components/data_transformation.py
# ═══════════════════════════════════════════════════════════════════
# Validated CSV → ColumnTransformer → numpy arrays (.npy)
# preprocessing.pkl save karta hai → PredictPipeline load karega
#
# NETWORKSECURITY SE FARQ:
# NetworkSecurity → KNNImputer only (sab numerical features)
# YAHAN → ColumnTransformer:
#   StandardScaler  → numerical cols (6)
#   OneHotEncoder   → categorical cols (4: brand, seller_type, fuel_type, transmission_type)
#   Output: 44 features (6 scaled + OHE expanded categoricals)
#
# FLOW:
# DataValidationArtifact (valid_train/test paths)
#       ↓ read_data()
# train_df, test_df
#       ↓ X/y split
# X_train (N×10), y_train (N,) — log1p selling_price
#       ↓ get_data_transformer_object()
# ColumnTransformer(StandardScaler + OHE)
#       ↓ fit_transform(X_train) ONLY — no leakage
# X_train_processed (N×44)
#       ↓ transform(X_test)
# X_test_processed  (M×44)
#       ↓ np.c_[X, y] → .npy
# train.npy (N×45), test.npy (M×45)
#       ↓ save preprocessing.pkl
# DataTransformationArtifact
###==============================================================

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from carprice.entity.config_entity import DataTransformationConfig
from carprice.entity.artifact_entity import (
    DataValidationArtifact,
    DataTransformationArtifact
)
from carprice.exception.exception import CarPriceException
from carprice.logging.logger import logging
from carprice.constants.training_pipeline import (
    TARGET_COLUMN,
    NUMERICAL_COLUMNS,
    CATEGORICAL_COLUMNS
)
from carprice.utils.main_utils.utils import (
    save_object,
    save_numpy_array_data
)


class DataTransformation:
    def __init__(self,
                 data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        """
        Parameters:
            data_validation_artifact : DataValidation ka output
                ├── valid_train_file_path
                └── valid_test_file_path

            data_transformation_config : config_entity.py se
                ├── transformed_train_file_path (.npy)
                ├── transformed_test_file_path  (.npy)
                └── transformed_object_file_path (.pkl)
        """
        try:
            self.data_validation_artifact   = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            logging.info("DataTransformation initialized")
        except Exception as e:
            raise CarPriceException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """
        CSV file padhta hai → DataFrame return karta hai.
        @staticmethod → self use nahi karta
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CarPriceException(e, sys)

    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        ColumnTransformer banata hai aur return karta hai.

        NETWORKSECURITY MEIN:
            Pipeline(KNNImputer) → sirf numerical features
        YAHAN:
            ColumnTransformer → mixed feature types
            ├── StandardScaler  → numerical_cols (6)
            └── OneHotEncoder   → categorical_cols (4)

        WHY COLUMNTRANSFORMER?
        har feature type alag treatment chahti hai:
        numerical → StandardScaler (mean=0, std=1)
        categorical → OHE (strings → 0/1 columns)
        ColumnTransformer → dono ek saath → ek object mein

        WHY STANDARDSCALER FOR ANN?
        ANN gradient descent pe train hota hai
        features same scale pe honi chahiye
        warna ek feature dominate karega → slow convergence

        Returns:
            ColumnTransformer : fitted hone ke baad transform karega
        """
        try:
            logging.info("Building ColumnTransformer")

            # ── Numerical pipeline ────────────────────────────────
            numerical_pipeline = Pipeline(steps=[
                ("scaler", StandardScaler())
                # StandardScaler → mean=0, std=1
                # vehicle_age (0-20) aur max_power (40-200) → same scale
            ])

            # ── Categorical pipeline ──────────────────────────────
            categorical_pipeline = Pipeline(steps=[
                ("encoder", OneHotEncoder(
                    sparse_output=False,
                    # sparse_output=False → dense numpy array return karo
                    # ANN ko dense array chahiye — sparse nahi
                    handle_unknown="ignore"
                    # ignore → test mein naya category aaye → zeros
                    # drop → test mein naya category → error
                ))
            ])

            # ── ColumnTransformer ──────────────────────────────────
            # NUMERICAL_COLUMNS = ["vehicle_age", "km_driven", ...] (constants se)
            # CATEGORICAL_COLUMNS = ["brand", "seller_type", ...] (constants se)
            preprocessor = ColumnTransformer(transformers=[
                ("num", numerical_pipeline, NUMERICAL_COLUMNS),
                ("cat", categorical_pipeline, CATEGORICAL_COLUMNS)
            ])

            logging.info(
                f"ColumnTransformer built:\n"
                f"  Numerical  ({len(NUMERICAL_COLUMNS)}): {NUMERICAL_COLUMNS}\n"
                f"  Categorical({len(CATEGORICAL_COLUMNS)}): {CATEGORICAL_COLUMNS}"
            )

            return preprocessor

        except Exception as e:
            raise CarPriceException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        DataTransformation ka main entry point.

        Returns:
            DataTransformationArtifact:
                ├── transformed_object_file_path → preprocessing.pkl
                ├── transformed_train_file_path  → train.npy (N×45)
                └── transformed_test_file_path   → test.npy  (M×45)
        """
        try:
            logging.info("=" * 50)
            logging.info("DataTransformation started")
            logging.info("=" * 50)

            # ── STEP 1: Data padhna ───────────────────────────────
            train_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_train_file_path
            )
            test_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_test_file_path
            )
            logging.info(
                f"Train: {train_df.shape} | Test: {test_df.shape}"
            )

            # ── STEP 2: X/y Split ─────────────────────────────────
            # TARGET_COLUMN = "selling_price" (constants se)
            # selling_price already log1p transformed hai (DataIngestion mein)
            # IMP: expm1() at prediction time → actual rupees
            X_train = train_df.drop(TARGET_COLUMN, axis=1)
            y_train = train_df[TARGET_COLUMN]

            X_test  = test_df.drop(TARGET_COLUMN, axis=1)
            y_test  = test_df[TARGET_COLUMN]

            logging.info(
                f"X_train: {X_train.shape} | y_train: {y_train.shape}\n"
                f"X_test : {X_test.shape}  | y_test : {y_test.shape}"
            )

            # ── STEP 3: Preprocessor banao ────────────────────────
            preprocessor = self.get_data_transformer_object()

            # ── STEP 4: Fit + Transform ───────────────────────────
            # IMP: fit_transform SIRF train pe
            # test pe sirf transform → no leakage
            X_train_processed = preprocessor.fit_transform(X_train)
            # fit → mean/std calculate from train
            # transform → apply to train

            X_test_processed  = preprocessor.transform(X_test)
            # sirf transform → train ki statistics use karo
            # test ke stats use karna → data leakage hoga

            logging.info(
                f"Processed shapes:\n"
                f"  X_train: {X_train_processed.shape}\n"
                f"  X_test : {X_test_processed.shape}"
            )
            # IMP: X_train_processed shape → (N, 44) expected
            # 6 numerical + OHE expanded categoricals

            # ── STEP 5: Features + Target Combine ────────────────
            # np.c_ → horizontally stack → last col = target
            # ModelTrainer mein: arr[:,:-1] = X, arr[:,-1] = y
            train_arr = np.c_[
                X_train_processed,
                np.array(y_train)
            ]
            test_arr = np.c_[
                X_test_processed,
                np.array(y_test)
            ]

            logging.info(
                f"Arrays:\n"
                f"  train_arr: {train_arr.shape}\n"
                f"  test_arr : {test_arr.shape}"
            )
            # train_arr shape → (N, 45) — 44 features + 1 target

            # ── STEP 6: Save Arrays ───────────────────────────────
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                train_arr
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                test_arr
            )
            logging.info("Numpy arrays saved")

            # ── STEP 7: Save Preprocessor ─────────────────────────
            # preprocessing.pkl → PredictPipeline load karega
            # fitted ColumnTransformer → transform new data
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor
            )
            logging.info(
                f"Preprocessor saved: "
                f"{self.data_transformation_config.transformed_object_file_path}"
            )

            # ── STEP 8: Artifact banao ────────────────────────────
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            logging.info(
                f"DataTransformation completed: {data_transformation_artifact}"
            )
            logging.info("=" * 50)

            return data_transformation_artifact

        except Exception as e:
            raise CarPriceException(e, sys)