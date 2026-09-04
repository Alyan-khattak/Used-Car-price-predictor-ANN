# ═══════════════════════════════════════════════════════════════════
# carprice/components/data_ingestion.py
# ═══════════════════════════════════════════════════════════════════
# CSV se data padhta hai → cleaning karta hai → train/test split
#
# NETWORKSECURITY SE FARQ:
# NetworkSecurity → MongoDB Atlas se data pull karta tha
# YAHAN → CSV file se directly padhte hain (Network_Data/ folder)
# MongoDB nahi — local CSV source
#
# FLOW:
# Car_Data/cardekho_dataset.csv
#       ↓ read_data_from_csv()
# raw DataFrame (15354 × 13)
#       ↓ apply_eda_cleaning()
# cleaned DataFrame (≈15000 × 11)
#       ↓ export_data_into_feature_store()
# Artifacts/.../feature_store/cardekho_dataset.csv
#       ↓ split_data_as_train_test()
# Artifacts/.../ingested/train.csv + test.csv
#       ↓ initiate_data_ingestion()
# DataIngestionArtifact
###==============================================================

import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from carprice.entity.config_entity import DataIngestionConfig
from carprice.entity.artifact_entity import DataIngestionArtifact

from carprice.exception.exception import CarPriceException
from carprice.logging.logger import logging

from carprice.constants.training_pipeline import(
    FUEL_TYPES_TO_DROP,
    SEATS_TO_DROP,
    SELLER_TYPE_MERGE,
    COLUMNS_TO_DROP,
    KM_DRIVEN_CAP_QUANTILE    
)



class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        DataIngestion initialize karta hai.

        Parameters:
            data_ingestion_config (DataIngestionConfig):
                config_entity.py se — sab paths yahan
                ├── feature_store_file_path
                ├── training_file_path
                ├── testing_file_path
                ├── train_test_split_ratio = 0.2
                └── random_state = 42
        """
        try:
            self.data_ingestion_config = data_ingestion_config
            logging.info("DataIngestion initialized with config")
        except Exception as e:
            raise CarPriceException(e, sys)

    def read_data_from_csv(self) -> pd.DataFrame:
        """
        Network_Data/ folder se raw CSV padhta hai.

        NETWORKSECURITY MEIN:
            MongoClient → collection.find() → pd.DataFrame()
        YAHAN:
            pd.read_csv() directly — MongoDB nahi

        Returns:
            pd.DataFrame : raw data (15354 × 13)
        """
        try:
            # IMP: CSV ka path → Network_Data/ folder mein hona chahiye
            raw_file_path = os.path.join(
                "CarPrice_Data",
                "cardekho_dataset.csv"
            )

            logging.info(f"Reading CSV from: {raw_file_path}")

            if not os.path.exists(raw_file_path):
                raise FileNotFoundError(
                    f"Dataset not found at: {raw_file_path}\n"
                    f"cardekho_dataset.csv → Network_Data/ folder mein rakho"
                )

            df = pd.read_csv(raw_file_path)
            logging.info(f"CSV loaded — Shape: {df.shape}")
            logging.info(f"Columns: {df.columns.tolist()}")

            return df

        except Exception as e:
            raise CarPriceException(e, sys)

    def apply_eda_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        EDA mein decide kiye gaye sab cleaning steps apply karta hai.
        Notebook se yahi decisions the — ab pipeline mein modular form mein.

        EDA DECISIONS (from notebook):
        1. Drop: Unnamed:0, car_name, model
        2. Drop: Electric, LPG fuel_type rows
        3. Merge: Trustmark Dealer → Dealer
        4. Drop: seats=0, seats=2
        5. Cap: km_driven at Q3 + 1.5×IQR
        6. Log1p: km_driven + selling_price

        Parameters:
            df (pd.DataFrame) : raw DataFrame

        Returns:
            pd.DataFrame : cleaned DataFrame
        """
        try:
            logging.info("Entered apply_eda_cleaning")
            df_clean = df.copy()
            # IMP: hamesha copy pe kaam karo — original preserve karo

            # ── STEP 1: Irrelevant columns drop karo ──────────────
            # Unnamed:0 → index column (useless)
            # car_name  → brand + model already separate columns hain
            # model     → too many unique values → OHE ke liye impractical
            cols_to_drop = [c for c in COLUMNS_TO_DROP
                            if c in df_clean.columns]
            df_clean = df_clean.drop(cols_to_drop, axis=1)
            logging.info(f"Dropped columns: {cols_to_drop} | Shape: {df_clean.shape}")

            # ── STEP 2: Rare fuel types drop karo ─────────────────
            # Electric → bahut kam rows, different features
            # LPG      → bahut kam rows
            # FUEL_TYPES_TO_DROP = ["Electric", "LPG"] (constants se)
            df_clean = df_clean[
                ~df_clean['fuel_type'].isin(FUEL_TYPES_TO_DROP)
            ]
            logging.info(f"Dropped fuel types: {FUEL_TYPES_TO_DROP} | Shape: {df_clean.shape}")

            # ── STEP 3: Trustmark Dealer → Dealer ─────────────────
            # Trustmark Dealer essentially Dealer hai
            # merge karo → 3 categories se 2
            # SELLER_TYPE_MERGE = {"Trustmark Dealer": "Dealer"} (constants se)
            df_clean['seller_type'] = df_clean['seller_type'].replace(
                SELLER_TYPE_MERGE
            )
            logging.info(f"Merged seller_type: {SELLER_TYPE_MERGE}")

            # ── STEP 4: Bad seats drop karo ───────────────────────
            # seats=0 → data error
            # seats=2 → bahut kam rows, niche sports cars
            # SEATS_TO_DROP = [0, 2] (constants se)
            df_clean = df_clean[~df_clean['seats'].isin(SEATS_TO_DROP)]
            logging.info(f"Dropped seats: {SEATS_TO_DROP} | Shape: {df_clean.shape}")

            # ── STEP 5: km_driven cap + log1p ─────────────────────
            # Cap at Q3 + 1.5×IQR → extreme outliers remove
            Q3  = df_clean['km_driven'].quantile(KM_DRIVEN_CAP_QUANTILE)
            IQR = Q3 - df_clean['km_driven'].quantile(0.25)
            cap = Q3 + 1.5 * IQR
            df_clean['km_driven'] = df_clean['km_driven'].clip(upper=cap)
            logging.info(f"km_driven capped at: {cap:.0f}")

            # log1p → skew reduce karo
            # log1p(0) = 0 → no issue with zero values
            df_clean['km_driven'] = np.log1p(df_clean['km_driven'])
            logging.info("km_driven log1p applied")

            # ── STEP 6: selling_price log1p ───────────────────────
            # Target column log transform
            # IMP: expm1() at prediction time → actual rupees mein convert
            df_clean['selling_price'] = np.log1p(df_clean['selling_price'])
            logging.info("selling_price log1p applied")

            # ── Drop duplicate log_price if exists ─────────────────
            # EDA notebook mein ek alag log_price col bani thi
            if 'log_price' in df_clean.columns:
                df_clean = df_clean.drop('log_price', axis=1)

            logging.info(f"Cleaning complete — Final shape: {df_clean.shape}")
            logging.info(f"Final columns: {df_clean.columns.tolist()}")

            return df_clean

        except Exception as e:
            raise CarPriceException(e, sys)

    def export_data_into_feature_store(self,
                                       dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Cleaned DataFrame ko feature store mein save karta hai.
        Raw data ka backup — split se pehle.

        Parameters:
            dataframe (pd.DataFrame) : cleaned DataFrame

        Returns:
            pd.DataFrame : same DataFrame (passthrough)

        STORES:
            Artifacts/timestamp/data_ingestion/feature_store/cardekho_dataset.csv
        """
        try:
            feature_store_file_path = \
                self.data_ingestion_config.feature_store_file_path

            logging.info(f"Saving to feature store: {feature_store_file_path}")

            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)

            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            logging.info(f"Feature store saved — Shape: {dataframe.shape}")

            return dataframe

        except Exception as e:
            raise CarPriceException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        """
        DataFrame ko 80/20 train/test split karta hai aur save karta hai.

        Parameters:
            dataframe (pd.DataFrame) : cleaned DataFrame

        STORES:
            Artifacts/timestamp/data_ingestion/ingested/train.csv
            Artifacts/timestamp/data_ingestion/ingested/test.csv
        """
        try:
            logging.info("Starting train/test split")

            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=self.data_ingestion_config.random_state
                # random_state=42 → reproducible split
            )

            logging.info(
                f"Split done — Train: {train_set.shape} | Test: {test_set.shape}"
            )

            # save karo
            ingested_dir = os.path.dirname(
                self.data_ingestion_config.training_file_path
            )
            os.makedirs(ingested_dir, exist_ok=True)

            train_set.to_csv(
                self.data_ingestion_config.training_file_path,
                index=False, header=True
            )
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path,
                index=False, header=True
            )

            logging.info(f"Train saved: {self.data_ingestion_config.training_file_path}")
            logging.info(f"Test saved:  {self.data_ingestion_config.testing_file_path}")

        except Exception as e:
            raise CarPriceException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        DataIngestion ka main entry point.
        Sab methods chain karta hai → DataIngestionArtifact return karta hai.

        Returns:
            DataIngestionArtifact:
                ├── train_file_path = "Artifacts/.../ingested/train.csv"
                └── test_file_path  = "Artifacts/.../ingested/test.csv"
        """
        try:
            logging.info("=" * 50)
            logging.info("DataIngestion started")
            logging.info("=" * 50)

            # STEP 1: CSV padhna
            df = self.read_data_from_csv()

            # STEP 2: EDA cleaning apply karo
            df_clean = self.apply_eda_cleaning(df)

            # STEP 3: Feature store mein save karo
            df_clean = self.export_data_into_feature_store(df_clean)

            # STEP 4: Train/test split + save
            self.split_data_as_train_test(df_clean)

            # STEP 5: Artifact banao
            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )

            logging.info(f"DataIngestion completed: {data_ingestion_artifact}")
            logging.info("=" * 50)

            return data_ingestion_artifact

        except Exception as e:
            raise CarPriceException(e, sys)