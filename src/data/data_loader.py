import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
from fredapi import Fred
import datetime
from typing import Dict, List, Tuple, Union, Optional

class MarketDataLoader:
    def __init__(self, api_key: Optional[str] = None):
        self.fred = Fred(api_key=api_key) if api_key else None
        
    def get_sp500_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        sp500 = yf.download('^GSPC', start=start_date, end=end_date)
        return sp500
    
    def get_macroeconomic_data(self, indicators: List[str], start_date: str, end_date: str) -> Dict[str, pd.Series]:
        if not self.fred:
            raise ValueError("API key for FRED is required to fetch macroeconomic data")
            
        data = {}
        for indicator in indicators:
            data[indicator] = self.fred.get_series(indicator, start_date, end_date)
        
        return data
    
    def get_sector_etfs(self, sectors: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        sector_data = {}
        for sector in sectors:
            sector_data[sector] = yf.download(sector, start=start_date, end=end_date)
        
        return sector_data
    
    def get_vix_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        vix = yf.download('^VIX', start=start_date, end=end_date)
        return vix
    
    def merge_datasets(self, market_data: pd.DataFrame, macro_data: Dict[str, pd.Series], 
                      fill_method: str = 'ffill') -> pd.DataFrame:
        mkt_df = market_data.copy()
        
        macro_df = pd.DataFrame(macro_data)
        
        if isinstance(mkt_df.columns, pd.MultiIndex):
            mkt_df.columns = [f"{col[0]}_{col[1]}" for col in mkt_df.columns]
        
        mkt_df.index = pd.to_datetime(mkt_df.index)
        macro_df.index = pd.to_datetime(macro_df.index)
        
        resampled_macro = macro_df.resample('B').ffill()
        
        merged = pd.concat([mkt_df, resampled_macro], axis=1)
        
        if fill_method == 'ffill':
            for col in macro_df.columns:
                if col in merged.columns and merged[col].notna().any():
                    merged[col] = merged[col].fillna(method='ffill')
        elif fill_method == 'bfill':
            for col in macro_df.columns:
                if col in merged.columns and merged[col].notna().any():
                    merged[col] = merged[col].fillna(method='bfill')
        
        for col in macro_df.columns:
            if col in merged.columns:
                nan_pct = merged[col].isna().mean()
                if nan_pct > 0.9:
                    merged = merged.drop(columns=[col])
        
        price_cols = [col for col in merged.columns if 'Close' in col]
        if price_cols:
            merged = merged.dropna(subset=price_cols)
        
        return merged

MACRO_INDICATORS = {
    'GDP': 'GDP',
    'UNRATE': 'UNRATE',
    'CPIAUCSL': 'CPIAUCSL',
    'FEDFUNDS': 'FEDFUNDS',
    'M2': 'M2',
    'INDPRO': 'INDPRO',
    'UMCSENT': 'UMCSENT',
    'HOUST': 'HOUST',
    'PAYEMS': 'PAYEMS',
    'PCE': 'PCE',
}

SECTOR_ETFS = {
    'XLK': 'Technology',
    'XLF': 'Financial',
    'XLV': 'Healthcare',
    'XLE': 'Energy',
    'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples',
    'XLI': 'Industrial',
    'XLB': 'Materials',
    'XLU': 'Utilities',
    'XLRE': 'Real Estate',
}