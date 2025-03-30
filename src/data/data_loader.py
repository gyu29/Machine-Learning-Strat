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
        yf.pdr_override()
        
    def get_sp500_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        sp500 = pdr.get_data_yahoo('^GSPC', start=start_date, end=end_date)
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
            sector_data[sector] = pdr.get_data_yahoo(sector, start=start_date, end=end_date)
        
        return sector_data
    
    def get_vix_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        vix = pdr.get_data_yahoo('^VIX', start=start_date, end=end_date)
        return vix
    
    def merge_datasets(self, market_data: pd.DataFrame, macro_data: Dict[str, pd.Series], 
                      fill_method: str = 'ffill') -> pd.DataFrame:
        macro_df = pd.DataFrame(macro_data)
        merged = pd.merge(market_data, macro_df, left_index=True, right_index=True, how='left')
        
        if fill_method == 'ffill':
            merged = merged.fillna(method='ffill')
        elif fill_method == 'bfill':
            merged = merged.fillna(method='bfill')
        
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