import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Union, Optional, Any
import joblib
import os
import datetime
import json

class BaseModel:
    def __init__(self, model_name: str, model_type: str, target_horizon: int):
        self.model_name = model_name
        self.model_type = model_type
        self.target_horizon = target_horizon
        self.model = None
        self.feature_importance = None
        self.training_history = None
        self.target_column = self._get_target_column()
        self.metadata = {
            'model_name': model_name,
            'model_type': model_type,
            'target_horizon': target_horizon,
            'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'last_trained': None,
            'last_evaluated': None,
            'performance_metrics': {}
        }
    
    def _get_target_column(self) -> str:
        if self.model_type == 'price':
            return f'Target_Price_{self.target_horizon}D'
        elif self.model_type == 'return':
            return f'Target_Return_{self.target_horizon}D'
        elif self.model_type == 'direction':
            return f'Target_Direction_{self.target_horizon}D'
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42,
                    feature_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data")
        
        if feature_columns is None:
            target_pattern = 'Target_'
            feature_columns = [col for col in df.columns if not col.startswith(target_pattern)]
        
        self.feature_columns = feature_columns
        self.metadata['feature_columns'] = feature_columns
        
        X = df[feature_columns].values
        y = df[self.target_column].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        raise NotImplementedError("Subclasses must implement train method")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(X_test)
        
        metrics = {}
        if self.model_type in ['price', 'return']:
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_test, y_pred)
            metrics['r2'] = r2_score(y_test, y_pred)
        else:
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['precision'] = precision_score(y_test, y_pred, zero_division=0)
            metrics['recall'] = recall_score(y_test, y_pred, zero_division=0)
            metrics['f1'] = f1_score(y_test, y_pred, zero_division=0)
        
        self.metadata['last_evaluated'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.metadata['performance_metrics'] = metrics
        
        return metrics
    
    def plot_feature_importance(self, feature_names: Optional[List[str]] = None, top_n: int = 20):
        if self.feature_importance is None:
            raise ValueError("Feature importance not available")
        
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(self.feature_importance))]
        
        indices = np.argsort(self.feature_importance)[-top_n:]
        plt.figure(figsize=(10, 8))
        plt.title(f'Top {top_n} Feature Importance for {self.model_name}')
        plt.barh(range(top_n), self.feature_importance[indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
    
    def save_model(self, directory: str):
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        model_path = os.path.join(directory, f"{self.model_name}.joblib")
        joblib.dump(self.model, model_path)
        
        metadata_path = os.path.join(directory, f"{self.model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=4)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, directory: str):
        model_path = os.path.join(directory, f"{self.model_name}.joblib")
        self.model = joblib.load(model_path)
        
        metadata_path = os.path.join(directory, f"{self.model_name}_metadata.json")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.model_type = self.metadata.get('model_type', self.model_type)
        self.target_horizon = self.metadata.get('target_horizon', self.target_horizon)
        self.feature_columns = self.metadata.get('feature_columns', None)
        
        print(f"Model loaded from {model_path}")
        
    def get_confidence_interval(self, X: np.ndarray, 
                               confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        y_pred = self.predict(X)
        
        if self.model_type in ['price', 'return']:
            std_dev = np.std(y_pred)
            z_score = 1.96
            
            lower_bound = y_pred - z_score * std_dev
            upper_bound = y_pred + z_score * std_dev
            
            return lower_bound, upper_bound
        else:
            return y_pred, y_pred