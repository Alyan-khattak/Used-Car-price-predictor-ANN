# ═══════════════════════════════════════════════════════════════════
# carprice/utils/dl_utils/model/estimator.py
# ═══════════════════════════════════════════════════════════════════
# Preprocessor + Keras model ko ek wrapper mein combine karta hai
#
# NETWORKSECURITY MEIN:
#   NetworkModel(preprocessor, sklearn_model)
#   predict() → preprocessor.transform() → model.predict()
#
# YAHAN (DL):
#   CarPriceModel(preprocessor, model_path)
#   predict() → preprocessor.transform() → keras model load → predict
#              → expm1() → actual rupees
#
# WHY ALAG ESTIMATOR?
# Keras model dill se serialize nahi hota properly
# isliye model path store karo — object nahi
# predict() pe load karo → predict → return
###==============================================================

import sys
import numpy as np
from carprice.exception.exception import CarPriceException
from carprice.logging.logger import logging


class CarPriceModel:
    def __init__(self, preprocessor, model_path: str):
        """
        Parameters:
            preprocessor : fitted ColumnTransformer
                           DataTransformation ne train kiya tha
                           → StandardScaler + OHE apply karega

            model_path   : str → path to .keras file
                           Keras model object nahi — path store karo
                           dill Keras objects properly serialize nahi karta
        """
        try:
            self.preprocessor = preprocessor
            self.model_path   = model_path
            logging.info(
                f"CarPriceModel initialized | model_path: {model_path}"
            )
        except Exception as e:
            raise CarPriceException(e, sys)

    def predict(self, x) -> np.ndarray:
        """
        Raw features → actual car price prediction (rupees mein)

        FLOW:
        raw DataFrame (N×10)
            ↓ preprocessor.transform()
        processed array (N×44)
            ↓ keras model.predict()
        log scale predictions (N,)
            ↓ np.expm1()
        actual rupees (N,)

        Parameters:
            x : pd.DataFrame ya numpy array (N×10 features)

        Returns:
            np.ndarray : predicted prices in actual rupees
                         e.g. [450000, 820000, 1200000]
        """
        try:
            import tensorflow as tf
            logging.info(f"Entered predict | input shape: {x.shape}")

            # ── STEP 1: Preprocess ────────────────────────────────
            x_transformed = self.preprocessor.transform(x)
            # ColumnTransformer:
            # numerical → StandardScaler
            # categorical → OHE
            # output → (N, 44) numpy array
            logging.info(f"Preprocessed shape: {x_transformed.shape}")

            # ── STEP 2: Load Keras model ──────────────────────────
            # IMP: har predict pe load karo
            # dill Keras objects serialize nahi karta
            # model path store kiya tha → yahan load karo
            model = tf.keras.models.load_model(self.model_path)
            logging.info(f"Model loaded from: {self.model_path}")

            # ── STEP 3: Predict ───────────────────────────────────
            y_pred_log = model.predict(x_transformed).flatten()
            # flatten() → (N,1) → (N,) 1D array

            # ── STEP 4: Reverse log1p → actual rupees ─────────────
            # IMP: selling_price log1p transform tha DataIngestion mein
            # expm1() → reverse karo → actual rupees
            y_pred_actual = np.expm1(y_pred_log)
            logging.info(
                f"Predictions (rupees): min={y_pred_actual.min():,.0f} "
                f"| max={y_pred_actual.max():,.0f}"
            )

            return y_pred_actual

        except Exception as e:
            raise CarPriceException(e, sys)