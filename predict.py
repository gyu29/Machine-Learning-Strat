#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.data_loader import MarketDataLoader, MACRO_INDICATORS
from src.models.random_forest_model import RandomForestModel
from src.config import FRED_API_KEY

def main():
    print("\n=== S&P 500 Buy/Sell Signal Generator ===\n")
    
    model_dir = 'models'
    price_model_path = os.path.join(model_dir, 'rf_price_20d.joblib')
    direction_model_path = os.path.join(model_dir, 'rf_direction_20d.joblib')
    
    if not (os.path.exists(price_model_path) and os.path.exists(direction_model_path)):
        print("Models not found. Please run the training first:")
        print("./run_model_demo.sh")
        return
    
    print("Loading latest market and macroeconomic data...")
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d') 
    data_loader = MarketDataLoader(api_key=FRED_API_KEY)
    
    try:
        sp500_data = data_loader.get_sp500_data(start_date, end_date)
        
        macro_data = data_loader.get_macroeconomic_data(
            list(MACRO_INDICATORS.keys()), 
            start_date, 
            end_date
        )
        
        df = data_loader.merge_datasets(sp500_data, macro_data)
        data_source = "market and macroeconomic"
    except Exception as e:
        print(f"Error fetching data: {e}")
        print("Attempting to use only market data for predictions...")
        try:
            sp500_data = data_loader.get_sp500_data(start_date, end_date)
            if isinstance(sp500_data.columns, pd.MultiIndex):
                sp500_data.columns = [f"{col[0]}_{col[1]}" for col in sp500_data.columns]
            df = sp500_data
            data_source = "market only"
        except Exception as e2:
            print(f"Error fetching market data: {e2}")
            print("Unable to fetch required data. Please check your internet connection.")
            return
    
    if len(df) == 0:
        print("No data available. Please check your internet connection and API key.")
        return
    
    print(f"Loaded {len(df)} days of {data_source} data.")
    print("\nData recency check:")
    latest_date = df.index.max()
    days_since_latest = (datetime.now().date() - latest_date.date()).days
    print(f"Latest data point: {latest_date.strftime('%Y-%m-%d')}")
    print(f"Days since latest data point: {days_since_latest}")
    market_columns = ['Open_^GSPC', 'High_^GSPC', 'Low_^GSPC', 'Close_^GSPC', 'Volume_^GSPC']
    macro_columns = [col for col in df.columns if col not in market_columns]
    
    if macro_columns:
        print("\nMacroeconomic indicators recency:")
        indicator_lags = {}
        
        typical_release_freq = {
            'GDP': 90,          
            'UNRATE': 30,       
            'CPIAUCSL': 30,     
            'FEDFUNDS': 7,      
            'M2': 7,            
            'INDPRO': 30,       
            'UMCSENT': 14,      
            'HOUST': 30,        
            'PAYEMS': 30,       
            'PCE': 30           
        }
        
        for col in macro_columns:
            last_valid = df[col].last_valid_index()
            
            if last_valid is not None:
                actual_days_since = (latest_date.date() - last_valid.date()).days
                indicator_lags[col] = actual_days_since
                
                typical_lag = typical_release_freq.get(col, 30)  
                
                if actual_days_since < typical_lag / 3:  
                    freshness = f"Current (unusually recent for {col}, typically ~{typical_lag} days)"
                else:
                    freshness = "Current" if actual_days_since <= 7 else "Recent" if actual_days_since <= 30 else "Lagged"
                
                print(f"  {col}: {last_valid.strftime('%Y-%m-%d')} ({actual_days_since} days lag) - {freshness}")
                print(f"     Typical release frequency: ~{typical_lag} days")
            else:
                print(f"  {col}: No valid data")
        
        current_indicators = [col for col, lag in indicator_lags.items() if lag <= 7]
        recent_indicators = [col for col, lag in indicator_lags.items() if 7 < lag <= 30]
        lagged_indicators = [col for col, lag in indicator_lags.items() if lag > 30]
        print("\nData recency summary:")
        print(f"  Market data: {days_since_latest} days old")
        print(f"  Current indicators (<=7 days lag): {len(current_indicators)}")
        print(f"  Recent indicators (8-30 days lag): {len(recent_indicators)}")
        print(f"  Lagged indicators (>30 days lag): {len(lagged_indicators)}")
        
        suspicious_indicators = 0
        for col in macro_columns:
            if col in indicator_lags and col in typical_release_freq:
                if indicator_lags[col] < typical_release_freq[col] / 3:
                    suspicious_indicators += 1
        
        if suspicious_indicators > len(macro_columns) / 2:
            print(f"\nNote: {suspicious_indicators} out of {len(macro_columns)} indicators are unusually current.")
            print("  This may indicate simulated/backfilled data or a data source issue.")
        
        data_quality_score = 1.0
        if len(macro_columns) > 0:
            current_ratio = len(current_indicators) / len(macro_columns)
            recent_ratio = len(recent_indicators) / len(macro_columns)
            lagged_ratio = len(lagged_indicators) / len(macro_columns)
            
            data_quality_score = current_ratio + (recent_ratio * 0.7) + (lagged_ratio * 0.4)
            
            print(f"  Overall data freshness score: {data_quality_score:.2f} (higher is better)")
    
    print("\nLoading prediction models...")
    price_model = RandomForestModel(
        model_name='rf_price_20d',
        model_type='price',
        target_horizon=20
    )
    price_model.load_model(model_dir)
    direction_model = RandomForestModel(
        model_name='rf_direction_20d',
        model_type='direction',
        target_horizon=20
    )
    
    direction_model.load_model(model_dir)
    print("Preparing features for prediction...")
    feature_columns = price_model.feature_columns
    print(f"Model expects {len(feature_columns)} features: {feature_columns}")
    print(f"Available columns in data: {df.columns.tolist()}")
    
    missing_features = [col for col in feature_columns if col not in df.columns]
    if len(missing_features) > len(feature_columns) / 2:
        print(f"Too many missing features ({len(missing_features)} out of {len(feature_columns)}).")
        print("The prediction may not be reliable. Consider retraining the model.")
        
    if missing_features:
        print(f"\nHandling {len(missing_features)} missing features:")
        
        try:
            hist_start = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
            hist_end = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
            print("Loading historical data for feature imputation...")
            hist_macro_data = data_loader.get_macroeconomic_data(
                [f for f in list(MACRO_INDICATORS.keys()) if MACRO_INDICATORS[f] in missing_features],
                hist_start, 
                hist_end
            )
            
            for col in missing_features:
                print(f"  Imputing {col}...")
                indicator_key = None
                for key, value in MACRO_INDICATORS.items():
                    if value == col:
                        indicator_key = key
                        break
                
                if indicator_key and indicator_key in hist_macro_data:
                    hist_values = hist_macro_data[indicator_key]
                    if len(hist_values) > 0:
                        recent_values = hist_values[-90:]
                        mean_value = recent_values.mean()
                        print(f"    Using historical mean: {mean_value:.4f}")
                        df[col] = mean_value
                        continue
                
                print(f"    No historical data available, using zero")
                df[col] = 0
                
        except Exception as e:
            print(f"Error loading historical data for imputation: {e}")
            print("Falling back to simple imputation...")
            
            for col in missing_features:
                print(f"  Adding placeholder zero values for: {col}")
                df[col] = 0
    
    df_features = df[feature_columns].copy()
    nan_columns = df_features.columns[df_features.isna().any()].tolist()
    if nan_columns:
        print("\nImputing NaN values in features:")
        for col in nan_columns:
            nan_count = df_features[col].isna().sum()
            print(f"  {col}: {nan_count} NaN values")
        
        imputer = SimpleImputer(strategy='mean')
        df_features_imputed = pd.DataFrame(
            imputer.fit_transform(df_features),
            columns=df_features.columns,
            index=df_features.index
        )
        df_features = df_features_imputed
    
    scaler = StandardScaler()
    latest_data = df.iloc[-1:].copy()
    scaled_features = scaler.fit_transform(df_features)
    latest_features = scaled_features[-1:, :]
    
    print("\nGenerating predictions...")
    price_pred = price_model.predict(latest_features)[0]
    direction_pred = direction_model.predict(latest_features)[0]
    direction_prob = direction_model.model.predict_proba(latest_features)[0, 1]
    
    current_price = latest_data['Close_^GSPC'].values[0] if 'Close_^GSPC' in latest_data.columns else None
    if current_price is None:
        for col in latest_data.columns:
            if 'Close' in col:
                current_price = latest_data[col].values[0]
                break
    
    unusual_prediction = False
    unusual_reason = []
    
    if current_price:
        predicted_return = ((price_pred / current_price) - 1) * 100
        
        if abs(predicted_return) > 30:
            unusual_prediction = True
            unusual_reason.append(f"Predicted return of {predicted_return:.2f}% is unusually large")
            
            if len(df) > 20:
                historical_returns = []
                for i in range(20, len(df)):
                    hist_return = ((df['Close_^GSPC'].iloc[i] / df['Close_^GSPC'].iloc[i-20]) - 1) * 100
                    historical_returns.append(hist_return)
                
                if historical_returns:
                    mean_return = np.mean(historical_returns)
                    std_return = np.std(historical_returns)
                    
                    max_reasonable_return = mean_return + 2.5 * std_return
                    min_reasonable_return = mean_return - 2.5 * std_return
                    
                    if predicted_return > max_reasonable_return:
                        adjusted_return = max_reasonable_return
                        adjusted_price = current_price * (1 + adjusted_return/100)
                        print(f"\nWarning: Prediction adjusted from ${price_pred:.2f} to ${adjusted_price:.2f}")
                        print(f"Original return of {predicted_return:.2f}% capped to {adjusted_return:.2f}%")
                        price_pred = adjusted_price
                        predicted_return = adjusted_return
                    elif predicted_return < min_reasonable_return:
                        adjusted_return = min_reasonable_return
                        adjusted_price = current_price * (1 + adjusted_return/100)
                        print(f"\nWarning: Prediction adjusted from ${price_pred:.2f} to ${adjusted_price:.2f}")
                        print(f"Original return of {predicted_return:.2f}% raised to {adjusted_return:.2f}%")
                        price_pred = adjusted_price
                        predicted_return = adjusted_return
    else:
        predicted_return = None
        unusual_prediction = True
        unusual_reason.append("Could not determine current price")
    
    signal = "BUY" if direction_pred == 1 else "SELL"
    confidence = direction_prob if direction_pred == 1 else (1 - direction_prob)
    confidence_str = f"{confidence * 100:.1f}%"
    data_quality_factor = 1.0
    
    if missing_features:
        missing_ratio = len(missing_features) / len(feature_columns)
        data_quality_factor *= (1 - missing_ratio * 0.5)
    
    if unusual_prediction:
        data_quality_factor *= 0.75
    
    if 'data_quality_score' in locals():
        print(f"\nData recency information: Overall freshness score {data_quality_score:.2f}")
    
    adjusted_confidence = confidence * data_quality_factor
    adjusted_confidence_str = f"{adjusted_confidence * 100:.1f}%"
    
    print("\n=== S&P 500 PREDICTION RESULTS ===")
    print(f"Current Date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"Current S&P 500 Price: ${current_price:.2f}")
    print(f"\nPrediction Horizon: 20 trading days (approx. 1 month)")
    print(f"Predicted Price: ${price_pred:.2f}")
    if predicted_return:
        print(f"Predicted 20-day Return: {predicted_return:.2f}%")
    print(f"\n20-day Direction Signal: {signal}")
    
    if data_quality_factor < 1.0:
        print(f"Raw Signal Confidence: {confidence_str}")
        print(f"Adjusted Signal Confidence: {adjusted_confidence_str} (adjusted for data quality)")
        
        if unusual_prediction:
            print("\nUnusual prediction detected:")
            for reason in unusual_reason:
                print(f"- {reason}")
    else:
        print(f"Signal Confidence: {confidence_str}")
    
    if 'suspicious_indicators' in locals() and suspicious_indicators > len(macro_columns) / 2:
        print("\nNote: Many macroeconomic indicators are updated more frequently than typical.")
        print("The predictions are not adjusted for this, but you should be aware when interpreting results.")
    
    print("\n=== RECOMMENDATION ===")
    
    if adjusted_confidence >= 0.75:
        strength = "STRONG"
    elif adjusted_confidence >= 0.6:
        strength = "MODERATE"
    else:
        strength = "WEAK"
    
    print(f"{strength} {signal} signal for the S&P 500 over the next 20 trading days.")
    
    if missing_features:
        print(f"\nNote: This prediction is based on incomplete data. {len(missing_features)} features were imputed.")
    
    if signal == "BUY":
        print("\nPositive outlook for the S&P 500. Consider implementing your buy strategy.")
    else:
        print("\nNegative outlook for the S&P 500. Consider implementing your sell/hedge strategy.")
    
    print("\nNote: This prediction is based on historical data and current market conditions.")
    print("Always do your own research and consider consulting a financial advisor before making investment decisions.")

if __name__ == "__main__":
    main()