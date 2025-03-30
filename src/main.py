import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.random_forest_model import RandomForestModel

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
    
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df['Close'], label='Price', alpha=0.7)
    plt.plot(df.index, df['MA_5'], label='MA_5', alpha=0.6)
    plt.plot(df.index, df['MA_20'], label='MA_20', alpha=0.6)
    plt.plot(df.index, df['MA_50'], label='MA_50', alpha=0.6)
    plt.title('Price and Moving Averages')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(df.index, df['RSI'], label='RSI', alpha=0.7)
    plt.plot(df.index, df['MACD'], label='MACD', alpha=0.7)
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
    
    # Create a mask for the upper triangle
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
    
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df['Close'].rolling(window=20).mean(), label='20-day Rolling Mean', alpha=0.7)
    plt.plot(df.index, df['Close'].rolling(window=20).std(), label='20-day Rolling Std', alpha=0.7)
    plt.title('Rolling Statistics of Price')
    plt.legend()
    plt.grid(True)
    
    returns = df['Close'].pct_change()
    plt.subplot(2, 1, 2)
    plt.plot(df.index, returns.rolling(window=20).mean(), label='20-day Rolling Mean', alpha=0.7)
    plt.plot(df.index, returns.rolling(window=20).std(), label='20-day Rolling Std', alpha=0.7)
    plt.title('Rolling Statistics of Returns')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('rolling_statistics.png')
    plt.close()

def main():
    df = create_synthetic_data()
    print(f"Data shape: {df.shape}")
    feature_columns = [col for col in df.columns if col.startswith('Feature_') or 
                      col in ['MA_5', 'MA_20', 'MA_50', 'RSI', 'MACD', 'MACD_signal']]
    
    print("Generating visualizations...")
    plot_time_series(df)
    plot_correlation_heatmap(df, feature_columns)
    plot_return_distributions(df)
    plot_rolling_statistics(df)
    print("Visualizations completed.")
    
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    price_model = RandomForestModel(
        model_name='rf_price_20d',
        model_type='price',
        target_horizon=20,
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    X_train, X_test, y_train, y_test = price_model.prepare_data(
        df, test_size=0.2, feature_columns=feature_columns
    )
    price_model.train(X_train, y_train)
    price_metrics = price_model.evaluate(X_test, y_test)
    for metric, value in price_metrics.items():
        print(f"  {metric}: {value:.4f}")
    direction_model = RandomForestModel(
        model_name='rf_direction_20d',
        model_type='direction',
        target_horizon=20,
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    X_train, X_test, y_train, y_test = direction_model.prepare_data(
        df, test_size=0.2, feature_columns=feature_columns
    )
    direction_model.train(X_train, y_train)
    direction_metrics = direction_model.evaluate(X_test, y_test)
    for metric, value in direction_metrics.items():
        print(f"  {metric}: {value:.4f}")
    plt.figure(figsize=(10, 6))
    importance = price_model.feature_importance
    indices = np.argsort(importance)[-10:]
    plt.title('Feature Importance for Price Prediction')
    plt.barh(range(len(indices)), importance[indices])
    plt.yticks(range(len(indices)), [feature_columns[i] for i in indices])
    plt.tight_layout()
    plt.savefig('price_feature_importance.png')
    os.makedirs('models', exist_ok=True)
    price_model.save_model('models')
    direction_model.save_model('models')
    price_preds = price_model.predict(X_test)
    direction_preds = direction_model.predict(X_test)
    lower_bound, upper_bound = price_model.get_confidence_interval(X_test)
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
    plt.tight_layout()
    plt.savefig('price_predictions.png')

if __name__ == "__main__":
    main()