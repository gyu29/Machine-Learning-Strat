import pandas as pd
import numpy as np
import datetime
from typing import Dict, List, Tuple, Union, Optional, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.models.base_model import BaseModel

class EnsembleModel(BaseModel):
    def __init__(self, model_name: str, model_type: str, target_horizon: int,
                base_models: List[BaseModel], weights: Optional[List[float]] = None,
                ensemble_method: str = 'weighted_average'):
        super().__init__(model_name, model_type, target_horizon)
        for model in base_models:
            if model.model_type != model_type:
                raise ValueError(f"Base model {model.model_name} has different model_type: {model.model_type} != {model_type}")
            if model.target_horizon != target_horizon:
                raise ValueError(f"Base model {model.model_name} has different target_horizon: {model.target_horizon} != {target_horizon}")
        self.base_models = base_models
        self.n_models = len(base_models)
        if weights is None:
            self.weights = np.ones(self.n_models) / self.n_models
        else:
            if len(weights) != self.n_models:
                raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({self.n_models})")
            self.weights = np.array(weights) / sum(weights)
        valid_methods = ['weighted_average', 'majority_vote', 'stacking']
        if ensemble_method not in valid_methods:
            raise ValueError(f"Ensemble method must be one of {valid_methods}")
        self.ensemble_method = ensemble_method
        self.meta_model = None
        self.metadata['base_model_names'] = [model.model_name for model in base_models]
        self.metadata['ensemble_method'] = ensemble_method
        self.metadata['weights'] = self.weights.tolist()
        self.metadata['model_class'] = 'Ensemble'
        self.model = self

    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        if self.ensemble_method in ['weighted_average', 'majority_vote']:
            self._optimize_weights(X_train, y_train)
        elif self.ensemble_method == 'stacking':
            self._train_stacking_model(X_train, y_train, **kwargs)
        self.metadata['last_trained'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.metadata['training_samples'] = len(X_train)
        self.metadata['weights'] = self.weights.tolist()
        print(f"Ensemble model {self.model_name} trained with {self.n_models} base models")

    def _optimize_weights(self, X_train: np.ndarray, y_train: np.ndarray):
        base_predictions = []
        for model in self.base_models:
            base_predictions.append(model.predict(X_train))
        base_predictions = np.array(base_predictions)
        if self.model_type in ['price', 'return']:
            errors = np.array([mean_absolute_error(y_train, pred) for pred in base_predictions])
            self.weights = 1 / errors
            self.weights = self.weights / np.sum(self.weights)
        else:
            accuracies = np.array([accuracy_score(y_train, (pred > 0.5).astype(int)) for pred in base_predictions])
            self.weights = accuracies
            self.weights = self.weights / np.sum(self.weights)
        self.metadata['weights'] = self.weights.tolist()
        self.metadata['base_model_metrics'] = {
            model.model_name: {'weight': weight} 
            for model, weight in zip(self.base_models, self.weights)
        }

    def _train_stacking_model(self, X_train: np.ndarray, y_train: np.ndarray, meta_model=None, **kwargs):
        base_predictions = []
        for model in self.base_models:
            base_predictions.append(model.predict(X_train))
        meta_features = np.column_stack(base_predictions)
        if meta_model is None:
            from sklearn.linear_model import Ridge, LogisticRegression
            if self.model_type in ['price', 'return']:
                meta_model = Ridge(alpha=1.0)
            else:
                meta_model = LogisticRegression(C=1.0, max_iter=1000)
        meta_model.fit(meta_features, y_train)
        self.meta_model = meta_model
        self.metadata['meta_model_type'] = str(type(meta_model).__name__)

    def predict(self, X: np.ndarray) -> np.ndarray:
        base_predictions = []
        for model in self.base_models:
            base_predictions.append(model.predict(X))
        base_predictions = np.array(base_predictions)
        if self.ensemble_method == 'weighted_average':
            final_predictions = np.sum(base_predictions * self.weights[:, np.newaxis], axis=0)
            if self.model_type == 'direction':
                final_predictions = (final_predictions > 0.5).astype(int)
        elif self.ensemble_method == 'majority_vote':
            if self.model_type != 'direction':
                raise ValueError("Majority vote only applicable for classification models")
            binary_predictions = (base_predictions > 0.5).astype(int)
            votes = np.dot(self.weights, binary_predictions)
            final_predictions = (votes > 0.5).astype(int)
        elif self.ensemble_method == 'stacking':
            if self.meta_model is None:
                raise ValueError("Meta-model not trained yet")
            meta_features = np.column_stack(base_predictions)
            final_predictions = self.meta_model.predict(meta_features)
        return final_predictions

    def get_confidence_interval(self, X: np.ndarray, 
                               confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        base_lower_bounds = []
        base_upper_bounds = []
        for model in self.base_models:
            lower, upper = model.get_confidence_interval(X, confidence_level)
            base_lower_bounds.append(lower)
            base_upper_bounds.append(upper)
        base_lower_bounds = np.array(base_lower_bounds)
        base_upper_bounds = np.array(base_upper_bounds)
        if self.ensemble_method == 'weighted_average':
            lower_bound = np.sum(base_lower_bounds * self.weights[:, np.newaxis], axis=0)
            upper_bound = np.sum(base_upper_bounds * self.weights[:, np.newaxis], axis=0)
        elif self.ensemble_method == 'majority_vote':
            lower_bound = np.median(base_lower_bounds, axis=0)
            upper_bound = np.median(base_upper_bounds, axis=0)
        elif self.ensemble_method == 'stacking':
            y_pred = self.predict(X)
            base_predictions = np.array([model.predict(X) for model in self.base_models])
            y_std = np.std(base_predictions, axis=0)
            alpha = 1 - confidence_level
            z_score = -np.percentile(np.random.normal(0, 1, 10000), alpha/2*100)
            lower_bound = y_pred - z_score * y_std
            upper_bound = y_pred + z_score * y_std
        return lower_bound, upper_bound