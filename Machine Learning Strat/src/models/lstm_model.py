import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import datetime
from typing import Dict, List, Tuple, Union, Optional, Any
from src.models.base_model import BaseModel
class LSTMModel(BaseModel):
    def __init__(self, model_name: str, model_type: str, target_horizon: int,
                sequence_length: int = 20, lstm_units: List[int] = [50],
                dense_units: List[int] = [20], dropout_rate: float = 0.2,
                learning_rate: float = 0.001, batch_size: int = 32,
                epochs: int = 100, early_stopping_patience: int = 10,
                random_state: int = 42):
        super().__init__(model_name, model_type, target_horizon)
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        self.sequence_length = sequence_length
        self.params = {
            'sequence_length': sequence_length,
            'lstm_units': lstm_units,
            'dense_units': dense_units,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs,
            'early_stopping_patience': early_stopping_patience,
            'random_state': random_state
        }
        self.metadata['model_params'] = self.params
        self.metadata['model_class'] = 'LSTM'
        self._initialize_model()
    def _initialize_model(self):
        model = Sequential()
        for i, units in enumerate(self.params['lstm_units']):
            if i == 0:
                model.add(LSTM(
                    units=units,
                    return_sequences=(i < len(self.params['lstm_units']) - 1),
                    input_shape=(self.sequence_length, None),
                ))
            else:
                model.add(LSTM(
                    units=units,
                    return_sequences=(i < len(self.params['lstm_units']) - 1),
                ))
            model.add(Dropout(self.params['dropout_rate']))
        for units in self.params['dense_units']:
            model.add(Dense(units=units, activation='relu'))
            model.add(Dropout(self.params['dropout_rate']))
        if self.model_type in ['price', 'return']:
            model.add(Dense(1, activation='linear'))
        else:
            model.add(Dense(1, activation='sigmoid'))
        if self.model_type in ['price', 'return']:
            model.compile(
                optimizer=Adam(learning_rate=self.params['learning_rate']),
                loss='mse',
                metrics=['mae']
            )
        else:
            model.compile(
                optimizer=Adam(learning_rate=self.params['learning_rate']),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        self.model = model
    def _prepare_sequences(self, X: np.ndarray, y: np.ndarray = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        n_samples = X.shape[0]
        n_features = X.shape[1]
        if n_samples <= self.sequence_length:
            raise ValueError(f"Not enough samples ({n_samples}) for sequence length ({self.sequence_length})")
        n_sequences = n_samples - self.sequence_length
        X_seq = np.zeros((n_sequences, self.sequence_length, n_features))
        for i in range(n_sequences):
            X_seq[i] = X[i:i+self.sequence_length]
        if y is not None:
            y_seq = y[self.sequence_length:]
            return X_seq, y_seq
        else:
            return X_seq
    def train(self, X_train: np.ndarray, y_train: np.ndarray, validation_split: float = 0.2, **kwargs):
        X_seq, y_seq = self._prepare_sequences(X_train, y_train)
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.params['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        )
        history = self.model.fit(
            X_seq, y_seq,
            batch_size=self.params['batch_size'],
            epochs=self.params['epochs'],
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        self.training_history = history.history
        self.metadata['last_trained'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.metadata['training_samples'] = len(X_seq)
        self.metadata['training_history'] = {
            'epochs': len(history.history['loss']),
            'final_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1])
        }
        print(f"Model {self.model_name} trained for {len(history.history['loss'])} epochs")
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained yet")
        X_seq = self._prepare_sequences(X)
        predictions = self.model.predict(X_seq)
        if self.model_type == 'direction':
            predictions = (predictions > 0.5).astype(int)
        return predictions.flatten()
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        X_seq, y_seq = self._prepare_sequences(X_test, y_test)
        keras_metrics = self.model.evaluate(X_seq, y_seq, verbose=0)
        y_pred = self.predict(X_test)
        y_test_aligned = y_test[self.sequence_length:]
        metrics = {}
        for i, metric_name in enumerate(self.model.metrics_names):
            metrics[f'keras_{metric_name}'] = keras_metrics[i]
        base_metrics = super().evaluate(X_seq, y_seq)
        metrics.update(base_metrics)
        return metrics
    def get_confidence_interval(self, X: np.ndarray, 
                               n_samples: int = 20,
                               confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise ValueError("Model not trained yet")
        X_seq = self._prepare_sequences(X)
        def mc_dropout_pred():
            return self.model(X_seq, training=True)
        predictions = [mc_dropout_pred().numpy() for _ in range(n_samples)]
        predictions = np.stack(predictions, axis=0)
        y_mean = np.mean(predictions, axis=0).flatten()
        y_std = np.std(predictions, axis=0).flatten()
        alpha = 1 - confidence_level
        z_score = -np.percentile(np.random.normal(0, 1, 10000), alpha/2*100)
        if self.model_type in ['price', 'return']:
            lower_bound = y_mean - z_score * y_std
            upper_bound = y_mean + z_score * y_std
        else:
            lower_bound = np.maximum(0, y_mean - z_score * y_std)
            upper_bound = np.minimum(1, y_mean + z_score * y_std)
        return lower_bound, upper_bound
