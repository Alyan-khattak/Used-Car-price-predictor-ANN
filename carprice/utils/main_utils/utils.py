# ═══════════════════════════════════════════════════════════════════
# carprice/utils/main_utils/utils.py
# ═══════════════════════════════════════════════════════════════════
# Common reusable utility functions — poore project mein import honti hain
#
# PEHLE TEEN PROJECTS MEIN:
#   save_object, load_object, evaluate_models (sklearn)
#
# YAHAN (DL project):
#   save_object       → preprocessor.pkl save (dill)
#   load_object       → preprocessor.pkl load (dill)
#   save_numpy_array  → .npy arrays save
#   load_numpy_array  → .npy arrays load
#   read_yaml_file    → schema.yaml padhna
#   write_yaml_file   → drift_report.yaml likhna
#   evaluate_ann      → KerasRegressor + GridSearchCV
###==============================================================

import os
import sys
import dill
import numpy as np
import yaml
from sklearn.model_selection import GridSearchCV

from carprice.exception.exception import CarPriceException
from carprice.logging.logger import logging


# ══════════════════════════════════════════════════════════════════
# FUNCTION 1: save_object
# ══════════════════════════════════════════════════════════════════
def save_object(file_path: str, obj) -> None:
    """
    Python object ko disk pe save karta hai (dill format).
    preprocessing.pkl save karne ke liye use hoga.

    Parameters:
        file_path (str) : jahan save karna hai
        obj             : jo save karna hai (ColumnTransformer etc.)

    WHY DILL not pickle?
    dill → complex sklearn objects (ColumnTransformer, Pipeline)
    handle kar sakta hai — pickle nahi kar sakta
    """
    try:
        logging.info(f"Entered save_object — saving to: {file_path}")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            # IMP: dill.dump(obj, file_obj) — obj pehle, file baad mein
            # BUG: dill.dump(file_obj, obj) → TypeError: file must have write

        logging.info(f"Object saved successfully: {file_path}")

    except Exception as e:
        raise CarPriceException(e, sys)


# ══════════════════════════════════════════════════════════════════
# FUNCTION 2: load_object
# ══════════════════════════════════════════════════════════════════
def load_object(file_path: str) -> object:
    """
    Disk se pkl file load karta hai.
    PredictPipeline preprocessing.pkl load karne ke liye use karega.

    Parameters:
        file_path (str) : pkl file ka path

    Returns:
        object : loaded object (ColumnTransformer etc.)
    """
    try:
        logging.info(f"Entered load_object — loading: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)

        logging.info(f"Object loaded successfully: {file_path}")
        return obj

    except Exception as e:
        raise CarPriceException(e, sys)


# ══════════════════════════════════════════════════════════════════
# FUNCTION 3: save_numpy_array_data
# ══════════════════════════════════════════════════════════════════
def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
    """
    Numpy array ko .npy binary format mein save karta hai.
    DataTransformation train.npy + test.npy save karne ke liye use karega.

    Parameters:
        file_path (str)        : .npy file ka path
        array     (np.ndarray) : save karna wala array

    WHY .npy?
    .npy = numpy ka native binary format
    CSV se fast load hota hai
    ANN ko directly numpy array chahiye
    """
    try:
        logging.info(f"Saving numpy array to: {file_path} | shape: {array.shape}")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)

        logging.info(f"Numpy array saved: {file_path}")

    except Exception as e:
        raise CarPriceException(e, sys)


# ══════════════════════════════════════════════════════════════════
# FUNCTION 4: load_numpy_array
# ══════════════════════════════════════════════════════════════════
def load_numpy_array(file_path: str) -> np.ndarray:
    """
    .npy file se numpy array load karta hai.
    ModelTrainer train.npy + test.npy load karne ke liye use karega.

    Parameters:
        file_path (str) : .npy file ka path

    Returns:
        np.ndarray : loaded array
    """
    try:
        logging.info(f"Loading numpy array from: {file_path}")

        with open(file_path, "rb") as file_obj:
            array = np.load(file_obj, allow_pickle=True)

        logging.info(f"Numpy array loaded | shape: {array.shape}")
        return array

    except Exception as e:
        raise CarPriceException(e, sys)


# ══════════════════════════════════════════════════════════════════
# FUNCTION 5: read_yaml_file
# ══════════════════════════════════════════════════════════════════
def read_yaml_file(file_path: str) -> dict:
    """
    YAML file padhta hai aur Python dict return karta hai.
    DataValidation schema.yaml padhne ke liye use karega.

    Parameters:
        file_path (str) : YAML file ka path
                          e.g. "data_schema/schema.yaml"

    Returns:
        dict : YAML content
               e.g. {"columns": [...], "numerical_columns": [...]}
    """
    try:
        logging.info(f"Reading YAML file: {file_path}")

        with open(file_path, "rb") as yaml_file:
            config = yaml.safe_load(yaml_file)

        logging.info(f"YAML file read successfully: {file_path}")
        return config

    except Exception as e:
        raise CarPriceException(e, sys)


# ══════════════════════════════════════════════════════════════════
# FUNCTION 6: write_yaml_file
# ══════════════════════════════════════════════════════════════════
def write_yaml_file(file_path: str, content: object,
                    replace: bool = False) -> None:
    """
    Python dict ko YAML file mein likhta hai.
    DataValidation drift_report.yaml likhne ke liye use karega.

    Parameters:
        file_path (str)    : jahan save karna hai
        content   (object) : jo likhna hai (dict → YAML)
        replace   (bool)   : True → purani file delete → naya banao
    """
    try:
        logging.info(f"Writing YAML file: {file_path}")

        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as file:
            yaml.dump(content, file)

        logging.info(f"YAML file written: {file_path}")

    except Exception as e:
        raise CarPriceException(e, sys)


# ══════════════════════════════════════════════════════════════════
# FUNCTION 7: evaluate_ann
# ══════════════════════════════════════════════════════════════════
def evaluate_ann(X_train, y_train,
                 param_grid: dict,
                 dropout_rate: float,
                 learning_rate: float,
                 batch_size: int) -> dict:
    """
    KerasRegressor + GridSearchCV se best ANN hyperparams dhundta hai.

    PEHLE TEEN PROJECTS MEIN:
        evaluate_models() → sklearn models → GridSearchCV → F1/R2 dict
        return {model_name: score}

    YAHAN (DL project):
        evaluate_ann() → KerasRegressor → GridSearchCV → best params
        return best_params dict (neurons, layers, epochs)

    WHY KerasRegressor?
    GridSearchCV sklearn ka hai — Keras models directly compatible nahi
    scikeras.KerasRegressor → Keras model ko sklearn API deta hai
    GridSearchCV isse normal sklearn model ki tarah treat karta hai

    Parameters:
        X_train       (array)  : training features (numpy)
        y_train       (array)  : training targets  (numpy)
        param_grid    (dict)   : search space
                                 e.g. {"neurons": [64,128,256],
                                        "layers": [1,2,3],
                                        "epochs": [50,100]}
        dropout_rate  (float)  : Dropout rate for ANN layers
        learning_rate (float)  : Adam optimizer learning rate
        batch_size    (int)    : training batch size

    Returns:
        dict : best params
               e.g. {"neurons": 128, "layers": 2, "epochs": 100}

    FLOW:
    X_train, y_train
        ↓ create_model(neurons, layers) → Sequential ANN
        ↓ KerasRegressor wraps it → sklearn compatible
        ↓ GridSearchCV(cv=3, scoring="r2")
        ↓ best_params_ → return
    """
    try:
        from scikeras.wrappers import KerasRegressor
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
        from tensorflow.keras.optimizers import Adam

        logging.info("Entered evaluate_ann — GridSearchCV for ANN hyperparams")
        logging.info(f"Param grid: {param_grid}")

        # ── create_model function ──────────────────────────────────
        def create_model(neurons: int = 64, layers: int = 1):
            """
            GridSearchCV yeh function call karega — neurons + layers vary karega

            Parameters:
                neurons (int) : units per hidden layer
                layers  (int) : number of hidden layers

            FLOW:
            input(n_features)
                → Dense(neurons, relu) + BN + Dropout  ← layer 1
                → Dense(neurons, relu) + BN + Dropout  ← layers 2..n
                → Dense(1)                              ← output (regression)
            """
            model = Sequential()

            # ── First hidden layer ─────────────────────────────────
            # IMP: input_shape sirf pehle layer mein
            model.add(Dense(neurons, activation='relu',
                            input_shape=(X_train.shape[1],)))
            model.add(BatchNormalization())
            # BatchNorm → training stable karta hai
            # activations normalize karta hai per batch
            model.add(Dropout(dropout_rate))
            # Dropout → overfitting reduce karta hai
            # dropout_rate=0.1 → 10% neurons randomly off during training

            # ── Additional hidden layers ───────────────────────────
            for _ in range(layers - 1):
                model.add(Dense(neurons, activation='relu'))
                model.add(BatchNormalization())
                model.add(Dropout(dropout_rate))

            # ── Output layer ───────────────────────────────────────
            model.add(Dense(1))
            # IMP: NO activation → regression output
            # sigmoid/softmax → classification ke liye hota hai
            # linear output → selling_price log scale mein

            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='mae',
                metrics=['mae']
            )
            return model

        # ── KerasRegressor ─────────────────────────────────────────
        # scikeras.KerasRegressor → sklearn compatible Keras wrapper
        # GridSearchCV isse normal sklearn estimator maanta hai
        keras_reg = KerasRegressor(
            model=create_model,
            batch_size=batch_size,
            verbose=0
            # epochs GridSearchCV inject karega param_grid se
        )

        # ── GridSearchCV ───────────────────────────────────────────
        # cv=3 → 3-fold cross validation (5 zyada slow hoga DL ke liye)
        # scoring="r2" → R² maximize karo
        # n_jobs=-1 → sab CPU cores use karo
        grid = GridSearchCV(
            estimator=keras_reg,
            param_grid=param_grid,
            cv=3,
            scoring="r2",
            n_jobs=-1,
            verbose=1
        )

        logging.info("GridSearchCV fitting started — yeh time lagega...")
        grid.fit(X_train, y_train)

        logging.info(f"GridSearchCV complete")
        logging.info(f"Best R²    : {grid.best_score_:.4f}")
        logging.info(f"Best params: {grid.best_params_}")

        return grid.best_params_
        # → {"neurons": 128, "layers": 2, "epochs": 100}
        # ModelTrainer in params se final model banayega

    except Exception as e:
        raise CarPriceException(e, sys)


# ─────────────────────────────────────────────────────────────────
# DRY RUN — evaluate_ann()
#
# param_grid = {
#     "neurons": [64, 128, 256],
#     "layers":  [1, 2, 3],
#     "epochs":  [50, 100]
# }
# → total combinations: 3 × 3 × 2 = 18
# → cv=3 → 18 × 3 = 54 fits
# → har fit ek ANN train karta hai
#
# best_params = evaluate_ann(X_train, y_train, param_grid, ...)
# → {"neurons": 128, "layers": 2, "epochs": 100}
#
# ModelTrainer:
# → final model 128-128-1 architecture se banayega
# → EarlyStopping(patience=10) se properly train karega
# ─────────────────────────────────────────────────────────────────