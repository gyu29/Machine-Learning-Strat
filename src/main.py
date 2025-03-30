import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.random_forest_model import RandomForestModel
from src.data.data_loader import MarketDataLoader, MACRO_INDICATORS
from src.config import FRED_API_KEY, DEFAULT_START_DATE, DEFAULT_RANDOM_STATE

def create_synthetic_data(n_samples=1000, n_features=20, noise=0.1):
    dates = pd.date_range(start='2018-01-01', periods=n_samples, freq='D')
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    true_beta = np.random.randn(n_features) / 10
    true_beta[0:5] = 0.2
    price = 100.0
    prices = []
    for i in range(n_samples):
        feature_effect = np.dot(X[i], true_beta)
        random_change = np.random.normal(0.0005, 0.01)
        price = price * (1 + feature_effect + random_change)
        prices.append(price)
    prices = np.array(prices)
    df = pd.DataFrame({'Close': prices}, index=dates)
    for i in range(n_features):
        df[f'Feature_{i+1}'] = X[:, i]
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    df['BB_std'] = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
    df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    for horizon in [5, 10, 20, 60]:
        df[f'Target_Price_{horizon}D'] = df['Close'].shift(-horizon)
        df[f'Target_Return_{horizon}D'] = df['Close'].pct_change(periods=horizon).shift(-horizon)
        df[f'Target_Direction_{horizon}D'] = (df[f'Target_Return_{horizon}D'] > 0).astype(int)
    df = df.dropna()
    return df

def plot_time_series(df):
    plt.figure(figsize=(15, 10))
    
    price_column = None
    if 'Close' in df.columns:
        price_column = 'Close'
    elif 'Adj Close' in df.columns:
        price_column = 'Adj Close'
    elif 'Close_^GSPC' in df.columns:
        price_column = 'Close_^GSPC'
    else:
        close_columns = [col for col in df.columns if 'Close' in col]
        if close_columns:
            price_column = close_columns[0]
    
    if not price_column:
        raise ValueError(f"Could not find a price column in the dataframe. Available columns: {df.columns.tolist()}")
    
    print(f"Using price column: {price_column}")
    
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df[price_column], label='Price', alpha=0.7)
    
    if 'MA_5' not in df.columns and price_column in df.columns:
        df['MA_5'] = df[price_column].rolling(window=5).mean()
    
    if 'MA_20' not in df.columns and price_column in df.columns:
        df['MA_20'] = df[price_column].rolling(window=20).mean()
    
    if 'MA_50' not in df.columns and price_column in df.columns:
        df['MA_50'] = df[price_column].rolling(window=50).mean()
    
    if 'MA_5' in df.columns:
        plt.plot(df.index, df['MA_5'], label='MA_5', alpha=0.6)
    
    if 'MA_20' in df.columns:
        plt.plot(df.index, df['MA_20'], label='MA_20', alpha=0.6)
    
    if 'MA_50' in df.columns:
        plt.plot(df.index, df['MA_50'], label='MA_50', alpha=0.6)
    
    plt.title('Price and Moving Averages')
    plt.legend()
    plt.grid(True)
    
    if 'RSI' not in df.columns and price_column in df.columns:
        delta = df[price_column].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
    
    if 'MACD' not in df.columns and 'MACD_signal' not in df.columns and price_column in df.columns:
        ema12 = df[price_column].ewm(span=12, adjust=False).mean()
        ema26 = df[price_column].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    plt.subplot(2, 1, 2)
    
    if 'RSI' in df.columns:
        plt.plot(df.index, df['RSI'], label='RSI', alpha=0.7)
    
    if 'MACD' in df.columns:
        plt.plot(df.index, df['MACD'], label='MACD', alpha=0.7)
    
    if 'MACD_signal' in df.columns:
        plt.plot(df.index, df['MACD_signal'], label='MACD Signal', alpha=0.7)
    
    plt.title('Technical Indicators')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('time_series_indicators.png')
    plt.close()

def plot_correlation_heatmap(df, feature_columns):
    plt.figure(figsize=(15, 12))
    
    technical_indicators = ['MA_5', 'MA_20', 'MA_50', 'RSI', 'MACD', 'MACD_signal']
    raw_features = [col for col in feature_columns if col.startswith('Feature_')]
    
    ordered_columns = technical_indicators + raw_features
    correlation_matrix = df[ordered_columns].corr()
    
    mask = np.triu(np.ones_like(correlation_matrix), k=1)
    
    sns.set(font_scale=1.2)
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='coolwarm', 
                center=0,
                fmt='.2f',
                square=True,
                cbar_kws={'shrink': .8},
                annot_kws={'size': 8})
    
    plt.title('Feature Correlation Heatmap', pad=20, size=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    sns.set(font_scale=1.0)

def plot_return_distributions(df):
    plt.figure(figsize=(15, 10))
    
    horizons = [5, 10, 20, 60]
    for i, horizon in enumerate(horizons, 1):
        plt.subplot(2, 2, i)
        returns = df[f'Target_Return_{horizon}D']
        sns.histplot(returns, bins=50, kde=True)
        plt.title(f'{horizon}-Day Returns Distribution')
        plt.xlabel('Return')
        plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('return_distributions.png')
    plt.close()

def plot_rolling_statistics(df):
    plt.figure(figsize=(15, 10))
    
    price_column = None
    if 'Close' in df.columns:
        price_column = 'Close'
    elif 'Adj Close' in df.columns:
        price_column = 'Adj Close'
    elif 'Close_^GSPC' in df.columns:
        price_column = 'Close_^GSPC'
    else:
        close_columns = [col for col in df.columns if 'Close' in col]
        if close_columns:
            price_column = close_columns[0]
    
    if not price_column:
        raise ValueError(f"Could not find a price column in the dataframe. Available columns: {df.columns.tolist()}")
    
    print(f"Using price column for rolling statistics: {price_column}")
    
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df[price_column].rolling(window=20).mean(), label='20-day Rolling Mean', alpha=0.7)
    plt.plot(df.index, df[price_column].rolling(window=20).std(), label='20-day Rolling Std', alpha=0.7)
    plt.title('Rolling Statistics of Price')
    plt.legend()
    plt.grid(True)
    
    returns = df[price_column].pct_change()
    plt.subplot(2, 1, 2)
    plt.plot(df.index, returns.rolling(window=20).mean(), label='20-day Rolling Mean', alpha=0.7)
    plt.plot(df.index, returns.rolling(window=20).std(), label='20-day Rolling Std', alpha=0.7)
    plt.title('Rolling Statistics of Returns')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('rolling_statistics.png')
    plt.close()

def create_target_variables(df, horizons=[5, 10, 20, 60]):
    result = df.copy()
    
    price_column = None
    if 'Close' in df.columns:
        price_column = 'Close'
    elif 'Adj Close' in df.columns:
        price_column = 'Adj Close'
    elif 'Close_^GSPC' in df.columns:
        price_column = 'Close_^GSPC'
    else:
        close_columns = [col for col in df.columns if 'Close' in col]
        if close_columns:
            price_column = close_columns[0]
    
    if not price_column:
        raise ValueError(f"Could not find a price column in the dataframe. Available columns: {df.columns.tolist()}")
    
    print(f"Using price column for target variables: {price_column}")
    
    for horizon in horizons:
        result[f'Target_Price_{horizon}D'] = result[price_column].shift(-horizon)
        result[f'Target_Return_{horizon}D'] = result[price_column].pct_change(periods=-horizon)
        result[f'Target_Direction_{horizon}D'] = (result[f'Target_Return_{horizon}D'] > 0).astype(int)
        
    return result

def main():
    # Initialize data loader with FRED API key
    data_loader = MarketDataLoader(api_key=FRED_API_KEY)
    
    # Either use real data if API key is provided or create synthetic data for demo
    try:
        if FRED_API_KEY:
            print("Using real market data with FRED API...")
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = DEFAULT_START_DATE
            
            # Get S&P 500 data
            sp500_data = data_loader.get_sp500_data(start_date, end_date)
            
            # Get macroeconomic indicators
            macro_data = data_loader.get_macroeconomic_data(list(MACRO_INDICATORS.keys()), start_date, end_date)
            
            # Merge data
            df = data_loader.merge_datasets(sp500_data, macro_data)
            print(f"Loaded real data with shape: {df.shape}")
        else:
            raise ValueError("No FRED API key provided")
    except Exception as e:
        print(f"Error loading real data: {e}")
        print("Falling back to synthetic data for demonstration...")
        df = create_synthetic_data()
        
    print(f"Data shape: {df.shape}")
    
    # Debug: Print column names
    print("Available columns:")
    for col in df.columns:
        print(f"  - {col}")
    
    # Create target variables for different time horizons
    df = create_target_variables(df, horizons=[5, 10, 20, 60])
    
    # Find the actual feature columns that are available
    feature_columns = []
    
    # Add any existing technical indicator columns
    tech_indicators = ['MA_5', 'MA_20', 'MA_50', 'RSI', 'MACD', 'MACD_signal']
    for indicator in tech_indicators:
        if indicator in df.columns:
            feature_columns.append(indicator)
    
    # Add feature columns that start with Feature_ (for synthetic data)
    feature_columns.extend([col for col in df.columns if col.startswith('Feature_')])
    
    # Add macroeconomic indicators
    for col in df.columns:
        if col in MACRO_INDICATORS.values() or (col not in feature_columns and col not in tech_indicators and not col.startswith('Target_')):
            feature_columns.append(col)
    
    print("Feature columns being used:")
    for col in feature_columns:
        print(f"  - {col}")
    
    print("Generating visualizations...")
    plot_time_series(df)
    
    if feature_columns:
        plot_correlation_heatmap(df, feature_columns)
    
    plot_return_distributions(df)
    plot_rolling_statistics(df)
    print("Visualizations completed.")
    
    # Clean data for modeling - drop rows with NaN values
    # Check for NaNs in the dataframe
    nan_counts = df.isna().sum()
    print("\nNaN counts in each column:")
    for col, count in nan_counts.items():
        if count > 0:
            print(f"  - {col}: {count}")
    
    # Drop rows with NaN values
    df_clean = df.dropna()
    print(f"\nShape after dropping NaN values: {df_clean.shape}")
    
    if df_clean.shape[0] < 100:
        print("Not enough data for model training after dropping NaN values.")
        return
    
    # Standardize the features for modeling
    scaler = StandardScaler()
    df_clean[feature_columns] = scaler.fit_transform(df_clean[feature_columns])
    
    # Set up price prediction model
    price_model = RandomForestModel(
        model_name='rf_price_20d',
        model_type='price',
        target_horizon=20,
        n_estimators=100,
        max_depth=10,
        random_state=DEFAULT_RANDOM_STATE
    )
    
    # Prepare data for price model
    X_train, X_test, y_train, y_test = price_model.prepare_data(
        df_clean, test_size=0.2, feature_columns=feature_columns,
        target_column='Target_Price_20D'
    )
    
    # Train price model
    print("\nTraining price prediction model...")
    price_model.train(X_train, y_train)
    
    # Evaluate price model
    price_metrics = price_model.evaluate(X_test, y_test)
    print("Price model metrics:")
    for metric, value in price_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Set up direction prediction model
    direction_model = RandomForestModel(
        model_name='rf_direction_20d',
        model_type='direction',
        target_horizon=20,
        n_estimators=100,
        max_depth=10,
        random_state=DEFAULT_RANDOM_STATE
    )
    
    # Prepare data for direction model
    X_train, X_test, y_train, y_test = direction_model.prepare_data(
        df_clean, test_size=0.2, feature_columns=feature_columns,
        target_column='Target_Direction_20D'
    )
    
    # Train direction model
    print("\nTraining direction prediction model...")
    direction_model.train(X_train, y_train)
    
    # Evaluate direction model
    direction_metrics = direction_model.evaluate(X_test, y_test)
    print("Direction model metrics:")
    for metric, value in direction_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    importance = price_model.feature_importance
    indices = np.argsort(importance)[-10:]
    plt.title('Feature Importance for Price Prediction')
    plt.barh(range(len(indices)), importance[indices])
    plt.yticks(range(len(indices)), [feature_columns[i] for i in indices])
    plt.tight_layout()
    plt.savefig('price_feature_importance.png')
    
    # Save models
    os.makedirs('models', exist_ok=True)
    price_model.save_model('models')
    direction_model.save_model('models')
    
    # Generate price predictions
    price_preds = price_model.predict(X_test)
    direction_preds = direction_model.predict(X_test)
    
    # Get confidence intervals
    lower_bound, upper_bound = price_model.get_confidence_interval(X_test)
    
    # Plot sample predictions
    sample_size = min(50, len(X_test))
    sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_test[sample_indices], 'b-', label='Actual')
    plt.plot(price_preds[sample_indices], 'r-', label='Predicted')
    plt.fill_between(
        range(sample_size),
        lower_bound[sample_indices],
        upper_bound[sample_indices],
        color='red',
        alpha=0.2,
        label='95% Confidence Interval'
    )
    plt.title('Price Prediction Model: Actual vs Predicted')
    plt.xlabel('Sample Index')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig('price_predictions.png')
    plt.close()

if __name__ == "__main__":
    main()