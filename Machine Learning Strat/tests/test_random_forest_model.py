import unittest
import numpy as np
import pandas as pd
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.random_forest_model import RandomForestModel

class TestRandomForestModel(unittest.TestCase):
    """Tests for the RandomForestModel class."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        
        # Create price data
        prices = np.random.normal(loc=100, scale=10, size=100)
        prices = np.cumsum(np.random.normal(loc=0, scale=1, size=100)) + 100
        
        # Create features
        X1 = np.random.normal(loc=0, scale=1, size=100)
        X2 = np.random.normal(loc=0, scale=1, size=100)
        X3 = np.random.normal(loc=0, scale=1, size=100)
        
        # Create target variables
        target_price_20D = np.roll(prices, -20)
        target_return_20D = np.roll(prices / prices - 1, -20)
        target_direction_20D = (target_return_20D > 0).astype(int)
        
        # Create DataFrame
        self.df = pd.DataFrame({
            'Close': prices,
            'X1': X1,
            'X2': X2,
            'X3': X3,
            'Target_Price_20D': target_price_20D,
            'Target_Return_20D': target_return_20D,
            'Target_Direction_20D': target_direction_20D
        }, index=dates)
        
        # Drop NaN values
        self.df = self.df.iloc[:-20].copy()  # Remove last 20 rows with NaN target values
        
        # Create features and target
        self.feature_columns = ['X1', 'X2', 'X3']
    
    def test_initialization(self):
        """Test model initialization."""
        # Initialize price model
        price_model = RandomForestModel(
            model_name='test_price_model',
            model_type='price',
            target_horizon=20,
            n_estimators=10,
            max_depth=5,
            random_state=42
        )
        
        # Check attributes
        self.assertEqual(price_model.model_name, 'test_price_model')
        self.assertEqual(price_model.model_type, 'price')
        self.assertEqual(price_model.target_horizon, 20)
        self.assertEqual(price_model.target_column, 'Target_Price_20D')
        self.assertEqual(price_model.params['n_estimators'], 10)
        self.assertEqual(price_model.params['max_depth'], 5)
        
        # Initialize direction model
        direction_model = RandomForestModel(
            model_name='test_direction_model',
            model_type='direction',
            target_horizon=20,
            n_estimators=10,
            random_state=42
        )
        
        # Check attributes
        self.assertEqual(direction_model.model_name, 'test_direction_model')
        self.assertEqual(direction_model.model_type, 'direction')
        self.assertEqual(direction_model.target_column, 'Target_Direction_20D')
    
    def test_training_and_prediction(self):
        """Test model training and prediction."""
        # Initialize model
        model = RandomForestModel(
            model_name='test_model',
            model_type='direction',
            target_horizon=20,
            n_estimators=10,
            random_state=42
        )
        
        # Prepare data
        X_train, X_test, y_train, y_test = model.prepare_data(
            self.df, 
            test_size=0.3, 
            feature_columns=self.feature_columns
        )
        
        # Train model
        model.train(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Check predictions
        self.assertEqual(len(y_pred), len(y_test))
        self.assertTrue(all(isinstance(p, (int, np.integer)) for p in y_pred))
        self.assertTrue(all(p in [0, 1] for p in y_pred))
        
        # Evaluate model
        metrics = model.evaluate(X_test, y_test)
        
        # Check metrics
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        
        # Check feature importance
        self.assertIsNotNone(model.feature_importance)
        self.assertEqual(len(model.feature_importance), len(self.feature_columns))
    
    def test_confidence_intervals(self):
        """Test confidence interval calculation."""
        # Initialize model
        model = RandomForestModel(
            model_name='test_model',
            model_type='price',
            target_horizon=20,
            n_estimators=10,
            random_state=42
        )
        
        # Prepare data
        X_train, X_test, y_train, y_test = model.prepare_data(
            self.df, 
            test_size=0.3, 
            feature_columns=self.feature_columns
        )
        
        # Train model
        model.train(X_train, y_train)
        
        # Get confidence intervals
        lower_bound, upper_bound = model.get_confidence_interval(X_test)
        
        # Check confidence intervals
        self.assertEqual(len(lower_bound), len(X_test))
        self.assertEqual(len(upper_bound), len(X_test))
        self.assertTrue(all(l <= u for l, u in zip(lower_bound, upper_bound)))
    
    def test_save_and_load(self):
        """Test model saving and loading."""
        import tempfile
        import shutil
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Initialize model
            model = RandomForestModel(
                model_name='test_save_model',
                model_type='direction',
                target_horizon=20,
                n_estimators=10,
                random_state=42
            )
            
            # Prepare data and train model
            X_train, X_test, y_train, y_test = model.prepare_data(
                self.df, 
                test_size=0.3, 
                feature_columns=self.feature_columns
            )
            model.train(X_train, y_train)
            
            # Save model
            model.save_model(temp_dir)
            
            # Check that files exist
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'test_save_model.joblib')))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'test_save_model_metadata.json')))
            
            # Create a new model instance
            loaded_model = RandomForestModel(
                model_name='test_save_model',
                model_type='direction',
                target_horizon=20
            )
            
            # Load the saved model
            loaded_model.load_model(temp_dir)
            
            # Make predictions with both models
            y_pred_original = model.predict(X_test)
            y_pred_loaded = loaded_model.predict(X_test)
            
            # Check that predictions are the same
            np.testing.assert_array_equal(y_pred_original, y_pred_loaded)
            
        finally:
            # Clean up
            shutil.rmtree(temp_dir)

if __name__ == '__main__':
    unittest.main() 