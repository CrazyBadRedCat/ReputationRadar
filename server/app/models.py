import logging
import threading
import os
from models.baseline import BaselineModel
from typing import Dict, Any, List, Tuple
from fastapi import UploadFile
from app.datasets_storage import DatasetStorage
from app.schemas import FitHyperparameters, BaselineHyperparameters, ModelDetails

MODEL_STORAGE_PATH = "/models"
DATASETS_STORAGE_PATH = "/datasets"


class ModelsDispatcher:
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.current_model_id: str = "baseline"
        self.training_threads: Dict[str, threading.Thread] = {}
        self.training_events: Dict[str, threading.Event] = {}
        self.training_status: Dict[str, str] = {}
        self.dataset_storage = DatasetStorage(DATASETS_STORAGE_PATH)
        self.logger = logging.getLogger("models_dispatcher")

    def list_models(self):
        """
        List available models with some metadata.
        """
        return [
            ModelDetails(id=model_id, status=self.training_status.get(model_id, "unknown"))
            for model_id in self.models
        ]

    def set_current_model(self, model_id: str):
        """
        Set the current model for further predictions.
        """
        if model_id not in self.models:
            raise KeyError(f"Model '{model_id}' does not exist")
        self.current_model_id = model_id

    def train(self, model_id: str, operation_id: str, hyperparameters: FitHyperparameters):
        """
        Train a model with given model_id and save it into loaded models
        """
        if model_id == "baseline":
            if hyperparameters and not isinstance(hyperparameters, BaselineHyperparameters):
                raise ValueError("Wrong hyperparameters type")
            return self._train_baseline(operation_id, hyperparameters)
        else:
            raise ValueError(f"Unknown model_id: '{model_id}'")

    def wait_for_training(self, model_id: str, timeout: int = 10):
        """
        Wait for the training to complete until the timeout happens.
        """
        if model_id not in self.training_events:
            raise ValueError(f"No training in progress for model_id {model_id}")

        completed = self.training_events[model_id].wait(timeout=timeout)
        if not completed:
            raise TimeoutError(f"Training for model_id {model_id} did not complete within {timeout} seconds")
        return {"status": self.training_status.get(model_id, "unknown")}

    def predict(self: str, X: List[str]) -> Tuple[str, int, List[float]]:
        """
        Preprocess the given text and make predictions using the given model
        """
        model = self.models.get(self.current_model_id, None)

        if model is None:
            raise KeyError(f"Model with ID {self.current_model_id} not found.")

        return model.predict(X)

    def save_all_models(self):
        """
        Save all trained models to disk.
        """
        self.logger.info("Saving models")
        os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)

        for _, model in self.models.items():
            model.save(MODEL_STORAGE_PATH)

    def load_models(self):
        """
        Load pretrained models from disk.
        """
        if not os.path.exists(MODEL_STORAGE_PATH) or not os.path.isdir(MODEL_STORAGE_PATH):
            self.logger.warning("Models path is missing or is not a directory")
            return

        for dir_name in os.listdir(MODEL_STORAGE_PATH):
            if dir_name.startswith("baseline_"):
                storage_path = os.path.join(MODEL_STORAGE_PATH, dir_name)
                fit_id = dir_name.replace("baseline_", "")
                model_to_load = BaselineModel(fit_id)
                model_to_load.load(storage_path)
                self.models["baseline"] = model_to_load
                self.training_status["baseline"] = "loaded"

    async def store_dataset(self, operation_id: str, file: UploadFile):
        await self.dataset_storage.store(file, f'{operation_id}.csv')

    def _train_baseline(self, operation_id: str, hyperparameters: BaselineHyperparameters):
        model_id = "baseline"
        """
        Train a model with given data using the BaselineModel and save it under a specific model_id.
        """
        def _train():
            try:
                self.training_status[model_id] = "in_progress"
                df = self.dataset_storage.load(f'{operation_id}.csv')
                model = BaselineModel(operation_id)
                model.fit(df, hyperparameters)
                self.models[model_id] = model
                self.training_status[model_id] = "completed"
            except Exception as e:
                self.training_status[model_id] = f"failed: {str(e)}"
                self.logger.error(f"During the model training, an error occured: {e}")
            finally:
                self.training_events[model_id].set()  # Notify that training is complete

        # Create a thread and event for training
        self.training_events[model_id] = threading.Event()
        training_thread = threading.Thread(target=_train)
        self.training_threads[model_id] = training_thread
        training_thread.start()
        return {"message": f"Training started for model {model_id}"}
