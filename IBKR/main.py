#!/usr/bin/env python3

import os
import sys
import logging
import argparse
import time
import datetime
from pathlib import Path

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).parent.parent))
from IBKR.config import *
from IBKR.ibkr_data import get_vanguard_sp500_data
from IBKR.ibkr_trading import TradingBot

def setup_logging():
    """Set up logging configuration"""
    log_file = LOGS_DIR / f"ibkr_bot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Reduce verbosity of other libraries
    logging.getLogger('ibapi').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

def check_environment():
    """Check if the environment is properly set up for trading"""
    logger = logging.getLogger(__name__)
    
    # Check if model directory exists
    if not MODEL_DIR.exists():
        logger.error(f"Model directory not found: {MODEL_DIR}")
        return False
    
    # Check for model files
    model_files = list(MODEL_DIR.glob("rf_*_20d.joblib"))
    if not model_files:
        logger.error(f"No model files found in {MODEL_DIR}")
        return False
    
    logger.info(f"Found {len(model_files)} model files: {[f.name for f in model_files]}")
    
    return True

def run_data_collection():
    """Run data collection only"""
    logger = logging.getLogger(__name__)
    logger.info("Starting data collection")
    
    try:
        data = get_vanguard_sp500_data()
        if data is not None:
            logger.info(f"Successfully collected {len(data)} rows of data")
            return True
        else:
            logger.error("Data collection failed")
            return False
    except Exception as e:
        logger.exception(f"Error in data collection: {e}")
        return False

def run_trading_bot(test_mode=False, check_interval=3600):
    """Run the trading bot"""
    logger = logging.getLogger(__name__)
    logger.info("Starting trading bot")
    
    bot = TradingBot()
    
    try:
        if test_mode:
            logger.info("Running in test mode (single execution)")
            success = bot.run_once()
            if success:
                logger.info("Test run completed successfully")
            else:
                logger.error("Test run failed")
        else:
            logger.info(f"Running in continuous mode with {check_interval}s interval")
            bot.run_continuously(check_interval=check_interval)
    except KeyboardInterrupt:
        logger.info("Trading bot interrupted by user")
    except Exception as e:
        logger.exception(f"Error in trading bot: {e}")
    finally:
        bot.stop()
        bot.disconnect()
        logger.info("Trading bot shut down")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='IBKR Trading Bot for Vanguard S&P 500 UCITS ETF')
    
    parser.add_argument('--test', action='store_true', help='Run in test mode (single execution)')
    parser.add_argument('--data-only', action='store_true', help='Only collect data, no trading')
    parser.add_argument('--interval', type=int, default=3600, help='Check interval in seconds (default: 3600)')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    logger = setup_logging()
    
    logger.info("=== IBKR Trading Bot for Vanguard S&P 500 UCITS ETF ===")
    logger.info(f"Bot configuration: Symbol={SYMBOL}, Exchange={EXCHANGE}, Currency={CURRENCY}")
    
    # Check environment before starting
    if not check_environment():
        logger.error("Environment check failed, exiting")
        return 1
    
    if args.data_only:
        logger.info("Running in data collection mode only")
        success = run_data_collection()
        return 0 if success else 1
    else:
        run_trading_bot(test_mode=args.test, check_interval=args.interval)
        return 0

if __name__ == "__main__":
    sys.exit(main()) 