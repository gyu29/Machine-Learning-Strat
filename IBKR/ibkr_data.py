import datetime
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.common import BarData
from ibapi.order import Order
import time
import threading
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from IBKR.config import *
from src.data.data_loader import MarketDataLoader, MACRO_INDICATORS

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "ibkr_data.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IBKRDataApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.nextValidOrderId = None
        self.contract_details = {}
        self.market_data = {}
        self.historical_data = {}
        self.errors = []
        self.is_connected = False
        self.data_ready_event = threading.Event()
        
    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.nextValidOrderId = orderId
        logger.info(f"Next Valid Order ID: {orderId}")
        self.is_connected = True
    
    def error(self, reqId, errorCode, errorString, *args):
        super().error(reqId, errorCode, errorString, *args)
        error_msg = f"Error {errorCode}: {errorString}"
        self.errors.append(error_msg)
        logger.error(error_msg)
        
        if errorCode in [1100, 1101, 1102, 2110]:
            self.is_connected = False
    
    def contractDetails(self, reqId, contractDetails):
        super().contractDetails(reqId, contractDetails)
        if reqId not in self.contract_details:
            self.contract_details[reqId] = []
        self.contract_details[reqId].append(contractDetails)
    
    def contractDetailsEnd(self, reqId):
        super().contractDetailsEnd(reqId)
        logger.info(f"ContractDetailsEnd. ReqId: {reqId}")
    
    def historicalData(self, reqId, bar):
        if reqId not in self.historical_data:
            self.historical_data[reqId] = []
        self.historical_data[reqId].append({
            "date": bar.date,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "average": bar.average,
            "barCount": bar.barCount
        })
    
    def historicalDataEnd(self, reqId, start, end):
        super().historicalDataEnd(reqId, start, end)
        logger.info(f"HistoricalDataEnd. ReqId: {reqId} from {start} to {end}")
        self.data_ready_event.set()
    
    def get_contract(self, symbol=SYMBOL, security_type=SECURITY_TYPE, 
                     exchange=EXCHANGE, currency=CURRENCY):
        contract = Contract()
        contract.symbol = symbol
        contract.secType = security_type
        contract.exchange = exchange
        contract.currency = currency
        return contract
    
    def request_historical_data(self, contract, duration="1 Y", bar_size="1 day",
                               what_to_show="ADJUSTED_LAST", use_rth=True):
        self.data_ready_event.clear()
        reqId = self.nextValidOrderId
        self.nextValidOrderId += 1
        
        end_datetime = ""
        self.reqHistoricalData(
            reqId, 
            contract, 
            end_datetime, 
            duration, 
            bar_size, 
            what_to_show, 
            use_rth, 
            1, 
            False, 
            []
        )
        
        timeout = 10
        if not self.data_ready_event.wait(timeout):
            logger.warning(f"Timeout waiting for historical data (reqId: {reqId})")
            
        return reqId
    
    def historical_data_to_df(self, reqId):
        if reqId not in self.historical_data or not self.historical_data[reqId]:
            logger.warning(f"No historical data for reqId: {reqId}")
            return pd.DataFrame()
        
        df = pd.DataFrame(self.historical_data[reqId])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)
        
        return df
    
    def get_etf_data(self, symbol=SYMBOL, duration="1 Y", bar_size="1 day"):
        contract = self.get_contract(symbol)
        reqId = self.request_historical_data(contract, duration, bar_size)
        return self.historical_data_to_df(reqId)
    
    def enrich_data(self, etf_data):
        df = etf_data.copy()
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        try:
            start_date = df.index.min().strftime('%Y-%m-%d')
            end_date = df.index.max().strftime('%Y-%m-%d')
            
            data_loader = MarketDataLoader(api_key=FRED_API_KEY)
            macro_data = data_loader.get_macroeconomic_data(
                list(MACRO_INDICATORS.keys()), 
                start_date, 
                end_date
            )
            
            df = data_loader.merge_datasets(df, macro_data)
        except Exception as e:
            logger.warning(f"Could not fetch macroeconomic data: {e}")
        
        return df
    
    def save_data(self, df, filename='etf_data.csv'):
        file_path = DATA_DIR / filename
        df.to_csv(file_path)
        logger.info(f"Data saved to {file_path}")
        return file_path

def connect_to_ibkr():
    app = IBKRDataApp()
    
    try:
        logger.info(f"Connecting to IBKR API at {TWS_HOST}:{TWS_PORT}")
        app.connect(TWS_HOST, TWS_PORT, CLIENT_ID)
        
        api_thread = threading.Thread(target=app.run, daemon=True)
        api_thread.start()
        
        timeout = 10
        start_time = time.time()
        while not app.is_connected and time.time() - start_time < timeout:
            time.sleep(0.1)
            
        if not app.is_connected:
            logger.error("Failed to connect to IBKR API")
            return None
        
        logger.info("Successfully connected to IBKR API")
        return app
    except Exception as e:
        logger.error(f"Error connecting to IBKR API: {e}")
        return None

def get_vanguard_sp500_data():
    app = connect_to_ibkr()
    if app is None:
        logger.error("Could not connect to IBKR API")
        return None
    
    try:
        logger.info(f"Fetching data for {SYMBOL} on {EXCHANGE}")
        etf_data = app.get_etf_data(SYMBOL)
        
        if etf_data.empty:
            logger.error(f"No data received for {SYMBOL}")
            app.disconnect()
            return None
        
        logger.info(f"Received {len(etf_data)} rows of data for {SYMBOL}")
        
        enriched_data = app.enrich_data(etf_data)
        
        app.save_data(enriched_data, f"{SYMBOL}_data.csv")
        
        app.disconnect()
        
        return enriched_data
    except Exception as e:
        logger.error(f"Error fetching ETF data: {e}")
        if app.is_connected:
            app.disconnect()
        return None

if __name__ == "__main__":
    data = get_vanguard_sp500_data()
    if data is not None:
        print(f"Successfully retrieved {len(data)} rows of data for {SYMBOL}")
    else:
        print("Failed to retrieve data") 