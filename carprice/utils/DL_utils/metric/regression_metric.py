# ═══════════════════════════════════════════════════════════════════
# carprice/utils/dl_utils/metric/regression_metric.py
# ═══════════════════════════════════════════════════════════════════
# Regression metrics calculate karta hai
# ModelTrainer train + test dono pe call karega
#
# NETWORKSECURITY MEIN:
#   get_classification_score() → f1, precision, recall
# YAHAN (regression):
#   get_regression_score() → mae, rmse, r2
###==============================================================

import sys
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from carprice.entity.artifact_entity import RegressionMetricArtifact
from carprice.exception.exception import CarPriceException
from carprice.logging.logger import logging


def get_regression_score(y_true, y_pred) -> RegressionMetricArtifact:
    """
    Regression metrics calculate karta hai.

    Parameters:
        y_true : actual values  (log scale)
        y_pred : predicted values (log scale)

    Returns:
        RegressionMetricArtifact:
            ├── mae      → Mean Absolute Error (log scale)
            ├── rmse     → Root Mean Squared Error (log scale)
            └── r2_score → R² (log scale)

    IMP: log scale pe metrics calculate karo
    expm1() baad mein karna → actual rupees mein convert
    """
    try:
        logging.info("Calculating regression metrics")

        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)

        logging.info(
            f"Metrics — MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}"
        )

        return RegressionMetricArtifact(
            mae=mae,
            rmse=rmse,
            r2_score=r2
        )

    except Exception as e:
        raise CarPriceException(e, sys)