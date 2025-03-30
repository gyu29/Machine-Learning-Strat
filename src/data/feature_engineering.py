import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple, Optional
import talib as ta

class FeatureEngineer:
    def __init__(self):
        pass
        
    def generate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result['SMA_20'] = ta.SMA(result['Close'].values, timeperiod=20)
        result['SMA_50'] = ta.SMA(result['Close'].values, timeperiod=50)
        result['SMA_200'] = ta.SMA(result['Close'].values, timeperiod=200)
        
        macd, macd_signal, macd_hist = ta.MACD(
            result['Close'].values, 
            fastperiod=12, 
            slowperiod=26, 
            signalperiod=9
        )
        result['MACD'] = macd
        result['MACD_Signal'] = macd_signal
        result['MACD_Hist'] = macd_hist
        
        result['RSI_14'] = ta.RSI(result['Close'].values, timeperiod=14)
        
        upper, middle, lower = ta.BBANDS(
            result['Close'].values,
            timeperiod=20,
            nbdevup=2,
            nbdevdn=2,
            matype=0
        )
        result['BB_Upper'] = upper
        result['BB_Middle'] = middle
        result['BB_Lower'] = lower
        
        result['ATR_14'] = ta.ATR(
            result['High'].values,
            result['Low'].values,
            result['Close'].values,
            timeperiod=14
        )
        
        slowk, slowd = ta.STOCH(
            result['High'].values,
            result['Low'].values,
            result['Close'].values,
            fastk_period=5,
            slowk_period=3,
            slowk_matype=0,
            slowd_period=3,
            slowd_matype=0
        )
        result['SlowK'] = slowk
        result['SlowD'] = slowd
        
        result['ADX_14'] = ta.ADX(
            result['High'].values,
            result['Low'].values,
            result['Close'].values,
            timeperiod=14
        )
        
        result['CCI_14'] = ta.CCI(
            result['High'].values,
            result['Low'].values,
            result['Close'].values,
            timeperiod=14
        )
        
        lookback = 120
        if len(result) >= lookback:
            high = result['Close'].rolling(window=lookback).max()
            low = result['Close'].rolling(window=lookback).min()
            diff = high - low
            
            result['Fib_0'] = low
            result['Fib_23.6'] = low + 0.236 * diff
            result['Fib_38.2'] = low + 0.382 * diff
            result['Fib_50'] = low + 0.5 * diff
            result['Fib_61.8'] = low + 0.618 * diff
            result['Fib_100'] = high
        
        result['Returns_1D'] = result['Close'].pct_change(1)
        result['Returns_5D'] = result['Close'].pct_change(5)
        result['Returns_10D'] = result['Close'].pct_change(10)
        result['Returns_20D'] = result['Close'].pct_change(20)
        
        result['Volatility_10D'] = result['Returns_1D'].rolling(window=10).std()
        result['Volatility_20D'] = result['Returns_1D'].rolling(window=20).std()
        
        result['Volume_SMA_10'] = result['Volume'].rolling(window=10).mean()
        result['Volume_SMA_20'] = result['Volume'].rolling(window=20).mean()
        result['Volume_Ratio'] = result['Volume'] / result['Volume_SMA_20']
        
        return result
    
    def generate_macroeconomic_features(self, df: pd.DataFrame, macro_data: Dict[str, pd.Series]) -> pd.DataFrame:
        result = df.copy()
        for indicator, series in macro_data.items():
            result[f'Macro_{indicator}'] = series
        
        for indicator, series in macro_data.items():
            if indicator in ['GDP', 'CPIAUCSL', 'M2', 'INDPRO', 'PCE']:
                result[f'Macro_{indicator}_Change'] = series.pct_change()
        
        for indicator, series in macro_data.items():
            if indicator in ['GDP', 'CPIAUCSL', 'INDPRO', 'PCE']:
                result[f'Macro_{indicator}_YoY'] = series.pct_change(12)
        
        if 'FEDFUNDS' in macro_data and 'CPIAUCSL' in macro_data:
            inflation = macro_data['CPIAUCSL'].pct_change(12) * 100
            result['Real_Rate'] = macro_data['FEDFUNDS'] - inflation
        
        return result
    
    def generate_relative_strength_features(self, df: pd.DataFrame, sector_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        result = df.copy()
        sp500_returns = result['Close'].pct_change()
        
        for sector, sector_df in sector_data.items():
            sector_returns = sector_df['Close'].pct_change()
            result[f'RS_{sector}'] = sp500_returns.divide(sector_returns)
            
            result[f'RS_{sector}_20D'] = (1 + sp500_returns).rolling(window=20).apply(
                lambda x: np.prod(x) - 1
            ).divide(
                (1 + sector_returns).rolling(window=20).apply(lambda x: np.prod(x) - 1)
            )
            
        return result
    
    def create_target_variables(self, df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
        result = df.copy()
        
        for horizon in horizons:
            result[f'Target_Price_{horizon}D'] = result['Close'].shift(-horizon)
            result[f'Target_Return_{horizon}D'] = result['Close'].pct_change(periods=-horizon)
            result[f'Target_Direction_{horizon}D'] = (result[f'Target_Return_{horizon}D'] > 0).astype(int)
            
        return result
    
    def prepare_dataset(self, df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
        result = df.copy()
        
        cols_to_drop = ['Adj Close']
        existing_cols = [col for col in cols_to_drop if col in result.columns]
        if existing_cols:
            result = result.drop(columns=existing_cols)
        
        if drop_na:
            result = result.dropna()
        else:
            result = result.fillna(method='ffill')
            result = result.fillna(0)
        
        return result