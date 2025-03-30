import datetime
import logging
import pandas as pd
import numpy as np
import threading
import time
import sys
import smtplib
import email.message
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.order_state import OrderState
from ibapi.common import OrderId, TickerId
from ibapi.utils import decimalMaxString, floatMaxString

# Add the parent directory to system path to import from src
sys.path.append(str(Path(__file__).parent.parent))
from IBKR.config import *
from IBKR.ibkr_data import connect_to_ibkr, IBKRDataApp
from src.models.random_forest_model import RandomForestModel

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "ibkr_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IBKRTradingApp(IBKRDataApp):
    """
    Application for IBKR trading based on ML model signals
    """
    def __init__(self):
        super().__init__()
        self.positions = {}
        self.account_summary = {}
        self.orders = {}
        self.order_status = {}
        self.executions = {}
        self.account_ready = threading.Event()
        self.order_filled_events = {}
        self.account_value = 0
        self.portfolio_value = 0
        self.buying_power = 0
        self.cash_balance = 0
        self.in_position = False
        self.current_position = None
        self.position_entry_price = None
        self.position_entry_date = None
        self.position_quantity = 0
        
    def position(self, account, contract, position, avgCost):
        """Callback for position information"""
        super().position(account, contract, position, avgCost)
        key = f"{contract.symbol}_{contract.exchange}_{contract.currency}"
        self.positions[key] = {
            'symbol': contract.symbol,
            'exchange': contract.exchange,
            'currency': contract.currency,
            'position': position,
            'avgCost': avgCost
        }
        
        # Update current position status for our target ETF
        if contract.symbol == SYMBOL and contract.exchange == EXCHANGE:
            if position != 0:
                self.in_position = True
                self.position_quantity = position
                self.position_entry_price = avgCost
            else:
                self.in_position = False
                self.position_quantity = 0
                self.position_entry_price = None
                
        logger.info(f"Position: {contract.symbol}, Qty: {position}, Avg Cost: {avgCost}")
    
    def positionEnd(self):
        """Callback for end of positions data"""
        super().positionEnd()
        logger.info("Position data received")
        
    def accountSummary(self, reqId, account, tag, value, currency):
        """Callback for account summary"""
        super().accountSummary(reqId, account, tag, value, currency)
        if tag not in self.account_summary:
            self.account_summary[tag] = value
        
        # Track key account values
        if tag == "TotalCashValue":
            self.cash_balance = float(value)
        elif tag == "NetLiquidation":
            self.account_value = float(value)
        elif tag == "BuyingPower":
            self.buying_power = float(value)
        elif tag == "GrossPositionValue":
            self.portfolio_value = float(value)
            
        logger.info(f"Account Summary: {tag} = {value} {currency}")
    
    def accountSummaryEnd(self, reqId):
        """Callback for end of account summary data"""
        super().accountSummaryEnd(reqId)
        logger.info("Account data received")
        self.account_ready.set()
    
    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, 
                   permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        """Callback for order status updates"""
        super().orderStatus(orderId, status, filled, remaining, avgFillPrice, 
                          permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice)
        
        self.order_status[orderId] = {
            'status': status,
            'filled': filled,
            'remaining': remaining,
            'avgFillPrice': avgFillPrice,
            'lastFillPrice': lastFillPrice
        }
        
        logger.info(f"Order Status: id={orderId}, status={status}, filled={filled}, remaining={remaining}")
        
        # If order is filled, signal any waiting threads
        if status in ["Filled", "Completed"] and orderId in self.order_filled_events:
            self.order_filled_events[orderId].set()
    
    def openOrder(self, orderId, contract, order, orderState):
        """Callback for open order information"""
        super().openOrder(orderId, contract, order, orderState)
        self.orders[orderId] = {
            'contract': contract,
            'order': order,
            'orderState': orderState
        }
        
    def execDetails(self, reqId, contract, execution):
        """Callback for execution details"""
        super().execDetails(reqId, contract, execution)
        if execution.orderId not in self.executions:
            self.executions[execution.orderId] = []
        self.executions[execution.orderId].append(execution)
        
        logger.info(f"Execution: orderId={execution.orderId}, time={execution.time}, side={execution.side}, "
                  f"shares={execution.shares}, price={execution.price}")
    
    def request_account_summary(self):
        """Request account summary information"""
        reqId = self.nextValidOrderId
        self.nextValidOrderId += 1
        self.account_ready.clear()
        self.reqAccountSummary(reqId, "All", "TotalCashValue,NetLiquidation,BuyingPower,GrossPositionValue")
        
        # Wait for account data
        timeout = 10  # seconds
        if not self.account_ready.wait(timeout):
            logger.warning("Timeout waiting for account data")
        
        return reqId
    
    def request_positions(self):
        """Request current positions"""
        self.positions = {}
        self.reqPositions()
        # Wait a moment for positions to be received
        time.sleep(2)
        
    def create_market_order(self, action, quantity, transmit=True):
        """Create a market order"""
        order = Order()
        order.action = action
        order.orderType = "MKT"
        order.totalQuantity = quantity
        order.transmit = transmit
        return order
    
    def create_limit_order(self, action, quantity, limit_price, transmit=True):
        """Create a limit order"""
        order = Order()
        order.action = action
        order.orderType = "LMT"
        order.totalQuantity = quantity
        order.lmtPrice = limit_price
        order.transmit = transmit
        return order
    
    def create_stop_order(self, action, quantity, stop_price, transmit=True):
        """Create a stop order"""
        order = Order()
        order.action = action
        order.orderType = "STP"
        order.totalQuantity = quantity
        order.auxPrice = stop_price
        order.transmit = transmit
        return order
    
    def place_order(self, contract, order, wait_for_fill=True, timeout=60):
        """Place an order and optionally wait for it to be filled"""
        orderId = self.nextValidOrderId
        self.nextValidOrderId += 1
        
        # Create event to wait for fill if needed
        if wait_for_fill:
            self.order_filled_events[orderId] = threading.Event()
        
        # Submit the order
        self.placeOrder(orderId, contract, order)
        logger.info(f"Order placed: ID={orderId}, Action={order.action}, Quantity={order.totalQuantity}, Type={order.orderType}")
        
        # Wait for the order to be filled if requested
        if wait_for_fill:
            if not self.order_filled_events[orderId].wait(timeout):
                logger.warning(f"Timeout waiting for order {orderId} to fill")
            del self.order_filled_events[orderId]
        
        return orderId
    
    def close_position(self, symbol=SYMBOL, exchange=EXCHANGE):
        """Close any existing position for the specified symbol"""
        if not self.in_position or self.position_quantity == 0:
            logger.info(f"No position to close for {symbol}")
            return None
        
        action = "SELL" if self.position_quantity > 0 else "BUY"
        quantity = abs(self.position_quantity)
        
        contract = self.get_contract(symbol, exchange=exchange)
        order = self.create_market_order(action, quantity)
        
        logger.info(f"Closing position: {action} {quantity} shares of {symbol}")
        order_id = self.place_order(contract, order)
        
        return order_id
    
    def enter_position(self, symbol=SYMBOL, exchange=EXCHANGE, quantity=None):
        """Enter a long position in the specified symbol"""
        if self.in_position:
            logger.info(f"Already in position for {symbol}, skipping entry")
            return None
        
        # Calculate position size if not specified
        if quantity is None:
            if self.account_value <= 0:
                logger.error("Cannot determine position size, account value not available")
                return None
            
            position_value = self.account_value * POSITION_SIZE
            
            # Get current price
            df = self.get_etf_data(symbol, duration="1 D", bar_size="1 min")
            if df.empty:
                logger.error("Could not get current price for position sizing")
                return None
                
            current_price = df['Close'].iloc[-1]
            quantity = int(position_value / current_price)
            
            if quantity <= 0:
                logger.error(f"Calculated quantity is invalid: {quantity}")
                return None
        
        contract = self.get_contract(symbol, exchange=exchange)
        order = self.create_market_order("BUY", quantity)
        
        logger.info(f"Entering position: BUY {quantity} shares of {symbol}")
        order_id = self.place_order(contract, order)
        
        return order_id
    
    def set_stop_loss(self, entry_price, symbol=SYMBOL, exchange=EXCHANGE):
        """Set a stop loss order for the current position"""
        if not self.in_position or self.position_quantity == 0:
            logger.warning("No position to set stop loss for")
            return None
        
        stop_price = round(entry_price * (1 - STOP_LOSS), 2)
        quantity = abs(self.position_quantity)
        
        contract = self.get_contract(symbol, exchange=exchange)
        order = self.create_stop_order("SELL", quantity, stop_price)
        
        logger.info(f"Setting stop loss: SELL {quantity} shares of {symbol} at {stop_price}")
        order_id = self.place_order(contract, order, wait_for_fill=False)
        
        return order_id
    
    def check_and_send_email(self, subject, body):
        """Send an email notification if enabled"""
        if not ENABLE_EMAIL_NOTIFICATIONS:
            return
            
        try:
            msg = email.message.EmailMessage()
            msg.set_content(body)
            msg['Subject'] = f"IBKR Trading Bot: {subject}"
            msg['From'] = EMAIL_SENDER
            msg['To'] = EMAIL_RECIPIENT
            
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email notification sent: {subject}")
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")

class TradingBot:
    """Trading bot that uses ML predictions to trade with IBKR"""
    def __init__(self, model_name="rf_direction_20d", model_type="direction", target_horizon=20):
        self.app = None
        self.model = None
        self.model_name = model_name
        self.model_type = model_type
        self.target_horizon = target_horizon
        self.is_running = False
        self.should_stop = False
        self.last_check_time = None
        
    def load_model(self):
        """Load the prediction model"""
        try:
            model = RandomForestModel(
                model_name=self.model_name,
                model_type=self.model_type,
                target_horizon=self.target_horizon
            )
            model.load_model(MODEL_DIR)
            logger.info(f"Loaded model: {self.model_name}")
            self.model = model
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def connect(self):
        """Connect to IBKR API"""
        try:
            self.app = IBKRTradingApp()
            self.app.connect(TWS_HOST, TWS_PORT, CLIENT_ID)
            
            # Start the API event thread
            api_thread = threading.Thread(target=self.app.run, daemon=True)
            api_thread.start()
            
            # Wait for connection to be established
            timeout = 10  # seconds
            start_time = time.time()
            while not self.app.is_connected and time.time() - start_time < timeout:
                time.sleep(0.1)
                
            if not self.app.is_connected:
                logger.error("Failed to connect to IBKR API")
                return False
            
            logger.info("Successfully connected to IBKR API")
            
            # Request account information
            self.app.request_account_summary()
            self.app.request_positions()
            
            return True
        except Exception as e:
            logger.error(f"Error connecting to IBKR: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from IBKR API"""
        if self.app and self.app.is_connected:
            self.app.disconnect()
            logger.info("Disconnected from IBKR API")
    
    def get_trading_data(self):
        """Get and process data for trading decisions"""
        try:
            # Fetch ETF data
            etf_data = self.app.get_etf_data(SYMBOL)
            if etf_data.empty:
                logger.error(f"No data received for {SYMBOL}")
                return None
            
            # Enrich with features needed for prediction
            enriched_data = self.app.enrich_data(etf_data)
            
            return enriched_data
        except Exception as e:
            logger.error(f"Error getting trading data: {e}")
            return None
    
    def generate_signals(self, data):
        """Generate trading signals using the ML model"""
        if self.model is None:
            logger.error("No model loaded for prediction")
            return None
        
        try:
            # Ensure all features needed for prediction are available
            feature_columns = self.model.feature_columns
            missing_features = [col for col in feature_columns if col not in data.columns]
            
            if missing_features:
                logger.warning(f"Missing features for prediction: {missing_features}")
                # Simple imputation for missing features
                for col in missing_features:
                    data[col] = 0
            
            # Get feature values for the latest data point
            X = data[feature_columns].iloc[-1:].values
            
            # Generate prediction
            prediction = self.model.predict(X)[0]
            
            if self.model_type == 'direction':
                signal = bool(prediction == 1)  # 1 for up, 0 for down or flat
                confidence = 0.75  # Default confidence
                
                # Try to get confidence interval if available
                try:
                    lower, upper = self.model.get_confidence_interval(X)
                    confidence = (upper[0] + 1 - lower[0]) / 2
                except:
                    pass
                
                logger.info(f"Signal generated: Direction={prediction}, Signal={'BUY' if signal else 'SELL'}, Confidence={confidence:.2f}")
                
                return {
                    'signal': signal,
                    'prediction': prediction,
                    'confidence': confidence,
                    'timestamp': data.index[-1]
                }
                
            elif self.model_type in ['price', 'return']:
                current_price = data['Close'].iloc[-1]
                
                if self.model_type == 'price':
                    predicted_price = prediction
                    expected_return = (predicted_price / current_price) - 1
                else:
                    expected_return = prediction
                    predicted_price = current_price * (1 + expected_return)
                
                signal = expected_return > ENTRY_THRESHOLD
                
                logger.info(f"Signal generated: CurrentPrice={current_price:.2f}, PredictedPrice={predicted_price:.2f}, " 
                          f"ExpectedReturn={expected_return:.2%}, Signal={'BUY' if signal else 'HOLD'}")
                
                return {
                    'signal': signal,
                    'prediction': prediction,
                    'current_price': current_price,
                    'predicted_price': predicted_price,
                    'expected_return': expected_return,
                    'timestamp': data.index[-1]
                }
                
            return None
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return None
    
    def check_exit_criteria(self, data):
        """Check if we should exit the current position"""
        if not self.app.in_position:
            return False
        
        try:
            current_price = data['Close'].iloc[-1]
            entry_price = self.app.position_entry_price
            
            if entry_price is None:
                logger.warning("No entry price available for exit check")
                return False
            
            # Calculate current return
            current_return = (current_price / entry_price) - 1
            
            # Check stop loss
            if current_return < -STOP_LOSS:
                logger.info(f"Stop loss triggered: Current return {current_return:.2%} < -{STOP_LOSS:.2%}")
                return True
            
            # Check take profit
            if current_return > TAKE_PROFIT:
                logger.info(f"Take profit triggered: Current return {current_return:.2%} > {TAKE_PROFIT:.2%}")
                return True
            
            # Check position duration
            if self.app.position_entry_date:
                days_held = (datetime.datetime.now() - self.app.position_entry_date).days
                if days_held > MAX_POSITION_DURATION:
                    logger.info(f"Max duration triggered: Position held for {days_held} days")
                    return True
            
            # Check for reversal signal
            signal_data = self.generate_signals(data)
            if signal_data and not signal_data['signal'] and signal_data['confidence'] > 0.6:
                logger.info("Exit signal from model")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking exit criteria: {e}")
            return False
    
    def execute_trading_logic(self):
        """Execute the main trading logic"""
        # Update account info and positions
        self.app.request_account_summary()
        self.app.request_positions()
        
        # Get the latest data
        data = self.get_trading_data()
        if data is None:
            logger.error("Could not get trading data")
            return
        
        # Check if we need to exit existing position
        if self.app.in_position:
            should_exit = self.check_exit_criteria(data)
            if should_exit:
                logger.info("Exit signal detected, closing position")
                order_id = self.app.close_position()
                if order_id:
                    self.app.check_and_send_email(
                        "Position Closed",
                        f"Closed position in {SYMBOL} at {data['Close'].iloc[-1]:.2f}"
                    )
                    # After closing, update positions
                    time.sleep(2)
                    self.app.request_positions()
        
        # Check if we should enter a new position
        if not self.app.in_position:
            signal_data = self.generate_signals(data)
            
            if signal_data and signal_data['signal']:
                # Check confidence/threshold for entry
                if ((self.model_type == 'direction' and signal_data['confidence'] > 0.5 + ENTRY_THRESHOLD) or
                    (self.model_type in ['price', 'return'] and signal_data['expected_return'] > ENTRY_THRESHOLD)):
                    
                    logger.info("Entry signal detected, opening position")
                    order_id = self.app.enter_position()
                    
                    if order_id:
                        # Get the entry price and set a stop loss
                        time.sleep(2)  # Wait for execution
                        self.app.request_positions()
                        
                        if self.app.in_position and self.app.position_entry_price:
                            self.app.set_stop_loss(self.app.position_entry_price)
                            
                            self.app.check_and_send_email(
                                "New Position",
                                f"Opened position in {SYMBOL} at {self.app.position_entry_price:.2f}"
                            )
    
    def run_once(self):
        """Run the trading logic once"""
        if not self.app or not self.app.is_connected:
            if not self.connect():
                return False
        
        if self.model is None:
            if not self.load_model():
                return False
        
        try:
            self.execute_trading_logic()
            self.last_check_time = datetime.datetime.now()
            return True
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            return False
    
    def run_continuously(self, check_interval=3600):
        """Run the trading bot continuously"""
        self.is_running = True
        self.should_stop = False
        
        if not self.connect():
            self.is_running = False
            return
        
        if not self.load_model():
            self.disconnect()
            self.is_running = False
            return
        
        logger.info(f"Trading bot started, checking every {check_interval} seconds")
        
        while not self.should_stop:
            try:
                current_time = datetime.datetime.now()
                current_day = current_time.strftime("%A")
                
                # Only trade on trading days during market hours
                if (current_day in TRADING_DAYS and
                    MARKET_OPEN_TIME <= current_time.strftime("%H:%M:%S") <= MARKET_CLOSE_TIME):
                    
                    logger.info("Market is open, executing trading logic")
                    self.execute_trading_logic()
                else:
                    logger.info("Market is closed, skipping trading check")
                
                self.last_check_time = current_time
                
                # Sleep until next check
                time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error in continuous trading loop: {e}")
                # Sleep briefly and retry
                time.sleep(60)
                
                # Reconnect if needed
                if not self.app or not self.app.is_connected:
                    logger.info("Attempting to reconnect...")
                    self.connect()
        
        self.disconnect()
        self.is_running = False
        logger.info("Trading bot stopped")
    
    def stop(self):
        """Stop the continuous trading loop"""
        self.should_stop = True
        logger.info("Stopping trading bot...")

def run_trading_bot():
    """Main function to run the trading bot"""
    bot = TradingBot()
    
    try:
        # Run once to test
        success = bot.run_once()
        if not success:
            logger.error("Initial test run failed, check logs for details")
            return
        
        # Run continuously
        bot.run_continuously()
    except KeyboardInterrupt:
        logger.info("Trading bot interrupted by user")
    finally:
        bot.stop()
        bot.disconnect()

if __name__ == "__main__":
    run_trading_bot() 