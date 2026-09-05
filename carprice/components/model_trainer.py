# ═══════════════════════════════════════════════════════════════════
# carprice/components/model_trainer.py
# ═══════════════════════════════════════════════════════════════════
# numpy arrays → ANN train → metrics → MLflow → model save
#
# NETWORKSECURITY SE FARQ:
# NetworkSecurity → GridSearchCV (sklearn) → best sklearn model
# YAHAN → KerasRegressor + GridSearchCV → best params
#          → best params se final ANN train → EarlyStopping
#          → TensorFlow/Keras model → .keras format save
#
# FLOW:
# DataTransformationArtifact (train.npy, test.npy, preprocessing.pkl)
#       ↓ load_numpy_array()
# train_arr, test_arr
#       ↓ X/y split (last col = target)
# X_train (N×44), y_train (N,)
#       ↓ get_best_params() → evaluate_ann() → GridSearchCV
# best_params {neurons, layers, epochs}
#       ↓ build_final_model(best_params)
# ANN architecture (best_neurons × best_layers → 1)
#       ↓ model.fit() + EarlyStopping
# trained model
#       ↓ get_regression_score() × 2 (train + test)
# train_metric, test_metric
#       ↓ track_mlflow()
# DagsHub experiment logged
#       ↓ CarPriceModel(preprocessor, model_path)
#       ↓ save final_model/
#       ↓ push_model_to_huggingface()
# ModelTrainerArtifact
###==============================================================

import os
import sys
import numpy as np
import mlflow
import dagshub

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from carprice.entity.config_entity import ModelTrainerConfig
from carprice.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    RegressionMetricArtifact
)
from carprice.exception.exception import CarPriceException
from carprice.logging.logger import logging
from carprice.constants.training_pipeline import (
    ANN_PARAM_GRID,
    ANN_DROPOUT_RATE,
    ANN_LEARNING_RATE,
    ANN_BATCH_SIZE,
    ANN_LOSS,
    ANN_EARLY_STOPPING_PATIENCE,
    ANN_EARLY_STOPPING_MONITOR,
    ANN_EARLY_STOPPING_RESTORE_BEST
)
from carprice.utils.main_utils.utils import (
    load_numpy_array,
    load_object,
    save_object,
    evaluate_ann
)
from carprice.utils.DL_utils.metric.regression_metric import get_regression_score
from carprice.utils.DL_utils.model.estimator import CarPriceModel
from carprice.cloud.hf_syncer import push_model_to_huggingface


class ModelTrainer:
    def __init__(self,
                 model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        """
        Parameters:
            model_trainer_config : config_entity.py se
                ├── trained_model_file_path
                ├── expected_r2_score = 0.80
                ├── overfitting_threshold = 0.05
                └── ANN hyperparams

            data_transformation_artifact : DataTransformation ka output
                ├── transformed_train_file_path → train.npy
                ├── transformed_test_file_path  → test.npy
                └── transformed_object_file_path → preprocessing.pkl
        """
        try:
            self.model_trainer_config         = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            logging.info("ModelTrainer initialized")
        except Exception as e:
            raise CarPriceException(e, sys)

    def track_mlflow(self, model_path: str,
                     metric: RegressionMetricArtifact,
                     prefix: str = ""):
        """
        MLflow mein metrics log karta hai → DagsHub pe sync hoga.

        IMP: dagshub.init() ANDAR hai — module level pe nahi
        Module level pe hoga → import hote hi auth maangega → Docker crash

        Parameters:
            model_path (str)                  : .keras file ka path
            metric     (RegressionMetricArtifact) : mae, rmse, r2
            prefix     (str)                  : "train_" ya "test_"
        """
        try:
            


            import dagshub
            dagshub.init(repo_owner='alyan-khattak',
             repo_name='Used-Car-price-predictor-ANN',
             mlflow=True)

            

            mlflow.set_experiment("CarPricePredictor")
            with mlflow.start_run():
                mlflow.log_metric(f"{prefix}mae",      metric.mae)
                mlflow.log_metric(f"{prefix}rmse",     metric.rmse)
                mlflow.log_metric(f"{prefix}r2_score", metric.r2_score)

                # Keras model log karo
                model = tf.keras.models.load_model(model_path)
                mlflow.tensorflow.log_model(model, "model")

            logging.info(
                f"MLflow logged [{prefix}] — "
                f"MAE: {metric.mae:.4f} | "
                f"RMSE: {metric.rmse:.4f} | "
                f"R²: {metric.r2_score:.4f}"
            )

        except Exception as e:
            raise CarPriceException(e, sys)

    def get_best_params(self, X_train, y_train) -> dict:
        """
        evaluate_ann() → KerasRegressor + GridSearchCV → best params

        Returns:
            dict: e.g. {"neurons": 128, "layers": 2, "epochs": 100}
        """
        try:
            logging.info("Starting hyperparameter search")

            best_params = evaluate_ann(
                X_train=X_train,
                y_train=y_train,
                param_grid=ANN_PARAM_GRID,
                dropout_rate=self.model_trainer_config.dropout_rate,
                learning_rate=self.model_trainer_config.learning_rate,
                batch_size=self.model_trainer_config.batch_size
            )

            logging.info(f"Best params found: {best_params}")
            return best_params

        except Exception as e:
            raise CarPriceException(e, sys)

    def build_final_model(self, input_dim: int,
                          neurons: int,
                          layers: int) -> Sequential:
        """
        Best params se final ANN architecture banata hai.

        NOTEBOOK ARCHITECTURE:
        44 → 256 → BN → Dropout
           → 128 → BN → Dropout
           → 64  → Dropout
           → 1 (linear)

        YAHAN:
        best_neurons × best_layers → dynamic architecture
        constants se dropout + learning rate

        Parameters:
            input_dim (int) : number of features (44)
            neurons   (int) : units per hidden layer (GridSearch best)
            layers    (int) : number of hidden layers (GridSearch best)

        Returns:
            Sequential : compiled Keras model
        """
        try:
            logging.info(
                f"Building ANN — input: {input_dim} | "
                f"neurons: {neurons} | layers: {layers}"
            )

            model = Sequential()

            # ── First hidden layer ─────────────────────────────────
            model.add(Dense(neurons, activation='relu',
                            input_shape=(input_dim,)))
            model.add(BatchNormalization())
            model.add(Dropout(self.model_trainer_config.dropout_rate))

            # ── Additional hidden layers ───────────────────────────
            for _ in range(layers - 1):
                model.add(Dense(neurons, activation='relu'))
                model.add(BatchNormalization())
                model.add(Dropout(self.model_trainer_config.dropout_rate))

            # ── Output layer ───────────────────────────────────────
            # NO activation → regression → linear output
            model.add(Dense(1))

            # ── Compile ────────────────────────────────────────────
            model.compile(
                optimizer=Adam(
                    learning_rate=self.model_trainer_config.learning_rate
                ),
                loss=self.model_trainer_config.loss,
                metrics=['mae']
            )

            model.summary()
            return model

        except Exception as e:
            raise CarPriceException(e, sys)

    def train_model(self, X_train, y_train,
                    X_test, y_test) -> ModelTrainerArtifact:
        """
        Full training pipeline:
        1. best params dhundho (GridSearch)
        2. final model banao
        3. EarlyStopping se train karo
        4. metrics calculate karo
        5. MLflow log karo
        6. model save karo
        7. HuggingFace push karo

        Returns:
            ModelTrainerArtifact
        """
        try:
            #############################################################
            # To Find Best Params

            # -> I am Commententing out this part b/c my system deont have GPU and can't perform gridsearch
            # -> I will add hardcoded values instead :: you can remove that and use this prt 
            #############################################################

            # ── STEP 1: Best params dhundho ───────────────────────
            # best_params = self.get_best_params(X_train, y_train)
            # best_neurons = best_params.get("model__neurons", 128)
            # best_layers  = best_params.get("model__layers",  2)
            # best_epochs  = best_params.get("epochs",         100)
            # IMP: scikeras prefix "model__" lagata hai params pe

            ###########################################################

# -------------------------------------------------------
#          HardCoded Prt
            best_neurons = 128
            best_layers  = 2
            best_epochs  = 100
#---------------------------------------------

            logging.info(
                f"Using — neurons: {best_neurons} | "
                f"layers: {best_layers} | epochs: {best_epochs}"
            )

            # ── STEP 2: Final model banao ─────────────────────────
            model = self.build_final_model(
                input_dim=X_train.shape[1],
                neurons=best_neurons,
                layers=best_layers
            )

            # ── STEP 3: EarlyStopping callback ────────────────────
            early_stopping = EarlyStopping(
                monitor=self.model_trainer_config.early_stopping_monitor,
                # monitor='val_loss' → validation loss pe watch karo
                patience=self.model_trainer_config.early_stopping_patience,
                # patience=10 → 10 epochs mein improve nahi → stop
                restore_best_weights=self.model_trainer_config.early_stopping_restore_best
                # True → best epoch ka model milega — not last epoch
            )

            # ── STEP 4: Train ─────────────────────────────────────
            logging.info("Training started")
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                # IMP: validation_data = test — not train
                epochs=best_epochs,
                batch_size=self.model_trainer_config.batch_size,
                callbacks=[early_stopping],
                verbose=1
            )
            logging.info(
                f"Training complete | "
                f"Epochs run: {len(history.history['loss'])}"
            )

            # ── STEP 5: Save model ────────────────────────────────
            model_dir = os.path.dirname(
                self.model_trainer_config.trained_model_file_path
            )
            os.makedirs(model_dir, exist_ok=True)

            model.save(self.model_trainer_config.trained_model_file_path)
            logging.info(
                f"Model saved: {self.model_trainer_config.trained_model_file_path}"
            )

            # ── STEP 6: Metrics calculate karo ───────────────────
            # log scale pe predict karo
            y_train_pred = model.predict(X_train).flatten()
            y_test_pred  = model.predict(X_test).flatten()

            train_metric = get_regression_score(y_train, y_train_pred)
            test_metric  = get_regression_score(y_test,  y_test_pred)

            logging.info(
                f"Train → MAE: {train_metric.mae:.4f} | "
                f"RMSE: {train_metric.rmse:.4f} | "
                f"R²: {train_metric.r2_score:.4f}"
            )
            logging.info(
                f"Test  → MAE: {test_metric.mae:.4f} | "
                f"RMSE: {test_metric.rmse:.4f} | "
                f"R²: {test_metric.r2_score:.4f}"
            )

            # ── STEP 7: R² threshold check ────────────────────────
            if test_metric.r2_score < self.model_trainer_config.expected_r2_score:
                raise Exception(
                    f"Model R² {test_metric.r2_score:.4f} < "
                    f"expected {self.model_trainer_config.expected_r2_score}"
                )

            # ── STEP 8: Overfitting check ─────────────────────────
            diff = abs(train_metric.r2_score - test_metric.r2_score)
            if diff > self.model_trainer_config.overfitting_threshold:
                logging.warning(
                    f"Overfitting detected | "
                    f"Train R²: {train_metric.r2_score:.4f} | "
                    f"Test R²: {test_metric.r2_score:.4f} | "
                    f"Diff: {diff:.4f} > {self.model_trainer_config.overfitting_threshold}"
                )

            # ── STEP 9: MLflow track karo ─────────────────────────
            self.track_mlflow(
                self.model_trainer_config.trained_model_file_path,
                train_metric,
                prefix="train_"
            )
            self.track_mlflow(
                self.model_trainer_config.trained_model_file_path,
                test_metric,
                prefix="test_"
            )

            # ── STEP 10: preprocessor load + CarPriceModel ────────
            preprocessor = load_object(
                self.data_transformation_artifact.transformed_object_file_path
            )

            # ── STEP 11: final_model/ mein save ───────────────────
            os.makedirs("final_model", exist_ok=True)

            # model .keras format mein
            model.save("final_model/model.keras")

            # preprocessor pkl mein
            save_object("final_model/preprocessor.pkl", preprocessor)

            logging.info("final_model/ saved")

            # ── STEP 12: HuggingFace push ─────────────────────────
            push_model_to_huggingface()

            # ── STEP 13: Artifact banao ───────────────────────────
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metric,
                test_metric_artifact=test_metric
            )

            logging.info(f"ModelTrainer completed: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise CarPriceException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        ModelTrainer ka main entry point.

        Returns:
            ModelTrainerArtifact
        """
        try:
            logging.info("=" * 50)
            logging.info("ModelTrainer started")
            logging.info("=" * 50)

            # ── STEP 1: Load numpy arrays ─────────────────────────
            train_arr = load_numpy_array(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr  = load_numpy_array(
                self.data_transformation_artifact.transformed_test_file_path
            )
            logging.info(
                f"Arrays loaded — "
                f"train: {train_arr.shape} | test: {test_arr.shape}"
            )

            # ── STEP 2: X/y split ─────────────────────────────────
            # last col = target (np.c_ ne yahan chipkaya tha)
            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]
            X_test  = test_arr[:, :-1]
            y_test  = test_arr[:, -1]

            logging.info(
                f"X_train: {X_train.shape} | y_train: {y_train.shape}\n"
                f"X_test : {X_test.shape}  | y_test : {y_test.shape}"
            )

            # ── STEP 3: Train ─────────────────────────────────────
            return self.train_model(X_train, y_train, X_test, y_test)

        except Exception as e:
            raise CarPriceException(e, sys)