#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.data_loader import MarketDataLoader, MACRO_INDICATORS, SECTOR_ETFS
from src.models.random_forest_model import RandomForestModel
from src.strategies.trading_strategy import TradingStrategy
from src.config import FRED_API_KEY

def main():
    print("S&P 500 Price Prediction and Trading Strategy Example")
    print("-" * 50)
    
    print(f"Initializing data loader with FRED API key: {FRED_API_KEY[:5]}...")
    data_loader = MarketDataLoader(api_key=FRED_API_KEY)
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
    
    print(f"Fetching S&P 500 data from {start_date} to {end_date}...")
    sp500_data = data_loader.get_sp500_data(start_date, end_date)
    
    print("\nS&P 500 Data Structure:")
    print(f"Shape: {sp500_data.shape}")
    print(f"Index type: {type(sp500_data.index)}")
    print(f"Index head: {sp500_data.index[:5]}")
    print(f"Columns: {sp500_data.columns.tolist()}")
    print(f"Sample data:\n{sp500_data.head(3)}")
    
    print(f"\nFetching macroeconomic indicators...")
    macro_data = data_loader.get_macroeconomic_data(
        list(MACRO_INDICATORS.keys())[:5],
        start_date, 
        end_date
    )
    
    print("\nMacro Data Structure:")
    macro_df = pd.DataFrame(macro_data)
    print(f"Shape: {macro_df.shape}")
    print(f"Index type: {type(macro_df.index)}")
    print(f"Index head: {macro_df.index[:5]}")
    print(f"Columns: {macro_df.columns.tolist()}")
    print(f"Sample data:\n{macro_df.head(3)}")
    
    def custom_merge(market_df, macro_df):
        print("\nPreparing to merge datasets...")
        
        mdf = market_df.copy()
        
        print("Fixing column names...")
        if isinstance(mdf.columns, pd.MultiIndex):
            print("Converting MultiIndex columns to simple columns...")
            mdf.columns = [f"{col[0]}_{col[1]}" for col in mdf.columns]
        
        print("Ensuring both dataframes have datetime index...")
        mdf.index = pd.to_datetime(mdf.index)
        macro_df.index = pd.to_datetime(macro_df.index)
        
        print("Resampling macro data to business day frequency...")
        resampled_macro = macro_df.resample('B').ffill()
        
        print("Merging datasets...")
        result = pd.concat([mdf, resampled_macro], axis=1)
        
        print("\nFilling missing values in a more controlled way...")
        
        for col in ['GDP', 'UNRATE', 'CPIAUCSL', 'FEDFUNDS', 'M2']:
            if col in result.columns:
                nan_count = result[col].isna().sum()
                print(f"Column {col} has {nan_count} NaN values before filling")
                
                if result[col].notna().any():
                    result[col] = result[col].fillna(method='ffill')
                    nan_count = result[col].isna().sum()
                    print(f"Column {col} has {nan_count} NaN values after filling")
        
        print("\nHandling remaining NaN values...")
        nan_counts = result.isna().sum(axis=1)
        print(f"Number of rows with NaN values: {(nan_counts > 0).sum()} out of {len(result)}")
        
        all_cols = result.columns.tolist()
        price_cols = [col for col in all_cols if any(x in col for x in ['Close', 'High', 'Low', 'Open', 'Volume'])]
        macro_cols = [col for col in all_cols if col not in price_cols]
        
        for col in macro_cols:
            nan_pct = result[col].isna().mean() * 100
            print(f"Column {col} has {nan_pct:.1f}% NaN values")
            if nan_pct > 50:
                print(f"Dropping column {col} due to too many NaN values")
                result = result.drop(columns=[col])
        
        print("\nChecking price data completeness...")
        price_nan_counts = result[price_cols[0]].isna().sum()
        print(f"Close price column has {price_nan_counts} NaN values")
        
        if price_nan_counts > 0:
            result = result.dropna(subset=price_cols)
            print(f"Dropped rows with missing close prices, now have {len(result)} rows")
        
        print("\nPreparing target columns for different time horizons...")
        
        close_column = None
        for col in result.columns:
            if isinstance(col, str) and 'Close' in col:
                close_column = col
                break
        
        if not close_column:
            if 'Close_^GSPC' in result.columns:
                close_column = 'Close_^GSPC'
            else:
                print(f"Available columns: {result.columns.tolist()}")
                raise ValueError("Could not find Close price column")
                
        print(f"Using {close_column} for target calculations")
        
        for horizon in [5, 10, 20, 60]:
            result[f'Target_Price_{horizon}D'] = result[close_column].shift(-horizon)
            result[f'Target_Return_{horizon}D'] = result[close_column].pct_change(periods=horizon).shift(-horizon)
            result[f'Target_Direction_{horizon}D'] = (result[f'Target_Return_{horizon}D'] > 0).astype(int)
        
        result_with_targets = result.dropna(subset=[f'Target_Direction_60D'])
        print(f"\nAfter preparing targets: {len(result_with_targets)} rows with complete target data")
        
        print("\nSample data with all features and targets:")
        print(result_with_targets.head(3))
        
        return result_with_targets
    
    print(f"\nMerging datasets with custom function...")
    try:
        df = custom_merge(sp500_data, macro_df)
        
        print(f"\nSuccessfully merged datasets!")
        print(f"Final dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
    except Exception as e:
        print(f"\nError during merging: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nExample complete! You can now use this data with the models and strategies.")

if __name__ == "__main__":
    main() 