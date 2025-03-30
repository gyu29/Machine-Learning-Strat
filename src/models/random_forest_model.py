import pandas as pd
import numpy as np
import os
import json
import joblib
import datetime
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, List, Tuple, Union, Optional, Any
from src.models.base_model import BaseModel
class RandomForestModel(BaseModel):
    def __init__(self, model_name, model_type, target_horizon, n_estimators=100, 
                 max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                 random_state=None, **kwargs):
        super().__init__(model_name, model_type, target_horizon)
        if self.model_type == 'price':
            self.target_column = f'Target_Price_{target_horizon}D'
        elif self.model_type == 'return':
            self.target_column = f'Target_Return_{target_horizon}D'
        elif self.model_type == 'direction':
            self.target_column = f'Target_Direction_{target_horizon}D'
        else:
            raise ValueError("model_type must be one of 'price', 'return', or 'direction'")
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'random_state': random_state,
            **kwargs
        }
        if self.model_type in ['price', 'return']:
            self.model = RandomForestRegressor(**self.params)
        else:
            self.model = RandomForestClassifier(**self.params)
        self.feature_importance = None
        self.feature_columns = None
        self.metadata['model_params'] = self.params
        self.metadata['model_class'] = 'RandomForest'
    def prepare_data(self, df, test_size=0.2, shuffle=True, feature_columns=None):
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in DataFrame")
        if feature_columns is None:
            self.feature_columns = [col for col in df.columns if col != self.target_column]
        else:
            self.feature_columns = feature_columns
        X = df[self.feature_columns].values
        y = df[self.target_column].values
        return train_test_split(X, y, test_size=test_size, shuffle=shuffle, 
                               random_state=self.params.get('random_state'))
    def train(self, X, y):
        self.model.fit(X, y)
        self.feature_importance = self.model.feature_importances_
        self.metadata['last_trained'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.metadata['training_samples'] = len(X)
        return self
    def predict(self, X):
        return self.model.predict(X)
    def tune_hyperparameters(self, X_train, y_train, param_grid=None, 
                           cv=5, scoring=None):
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        if scoring is None:
            if self.model_type in ['price', 'return']:
                scoring = 'neg_mean_squared_error'
            else:
                scoring = 'f1'
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        self.params.update(grid_search.best_params_)
        self.metadata['model_params'] = self.params
        self.metadata['tuning_results'] = {
            'best_score': grid_search.best_score_,
            'best_params': grid_search.best_params_,
            'cv': cv,
            'scoring': scoring
        }
        return grid_search.best_params_
    def get_confidence_interval(self, X, confidence=0.95):
        if self.model_type == 'direction':
            raise ValueError("Confidence intervals are not available for classification models")
        preds = np.array([tree.predict(X) for tree in self.model.estimators_])
        alpha = (1 - confidence) / 2
        lower_bound = np.percentile(preds, alpha * 100, axis=0)
        upper_bound = np.percentile(preds, (1 - alpha) * 100, axis=0)
        return lower_bound, upper_bound
    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        if self.model_type in ['price', 'return']:
            metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred)
            }
        else:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='binary'),
                'recall': recall_score(y_true, y_pred, average='binary'),
                'f1': f1_score(y_true, y_pred, average='binary')
            }
        return metrics
    def save_model(self, directory):
        os.makedirs(directory, exist_ok=True)
        model_path = os.path.join(directory, f"{self.model_name}.joblib")
        joblib.dump(self.model, model_path)
        metadata = {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'target_horizon': self.target_horizon,
            'target_column': self.target_column,
            'params': self.params,
            'feature_columns': self.feature_columns,
            'feature_importance': self.feature_importance.tolist() if self.feature_importance is not None else None
        }
        metadata_path = os.path.join(directory, f"{self.model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
    def load_model(self, directory):
        model_path = os.path.join(directory, f"{self.model_name}.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = joblib.load(model_path)
        metadata_path = os.path.join(directory, f"{self.model_name}_metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        self.model_type = metadata['model_type']
        self.target_horizon = metadata['target_horizon']
        self.target_column = metadata['target_column']
        self.params = metadata['params']
        self.feature_columns = metadata['feature_columns']
        if metadata['feature_importance'] is not None:
            self.feature_importance = np.array(metadata['feature_importance'])