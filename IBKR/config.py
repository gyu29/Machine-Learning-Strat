import os
from pathlib import Path

# IBKR API connection settings
TWS_HOST = "127.0.0.1"  # TWS/IB Gateway IP address
TWS_PORT = 7497  # 7497 for TWS Paper Trading, 7496 for TWS Live, 4002 for IB Gateway Paper, 4001 for IB Gateway Live
CLIENT_ID = 1  # Unique client ID

# Vanguard S&P 500 UCITS ETF details
SYMBOL = "VUSA"  # Vanguard S&P 500 UCITS ETF symbol
EXCHANGE = "LSE"  # London Stock Exchange
CURRENCY = "GBP"  # Currency for the ETF
SECURITY_TYPE = "STK"  # Security type (STK for stocks and ETFs)

# Trading parameters
ENTRY_THRESHOLD = 0.05  # Signal threshold for entry
EXIT_THRESHOLD = 0.05   # Signal threshold for exit
STOP_LOSS = 0.10        # Stop-loss percentage
TAKE_PROFIT = 0.20      # Take-profit percentage
MAX_POSITION_DURATION = 60  # Maximum holding period in days
MIN_POSITION_DURATION = 20  # Minimum holding period in days
POSITION_SIZE = 0.10    # Percentage of account to use per trade (0.10 = 10%)
MAX_DRAWDOWN = 0.15     # Maximum drawdown threshold
ENABLE_RISK_ADJUSTMENT = True  # Dynamic position sizing based on volatility

# File paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = BASE_DIR.parent / "models"
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"

# Ensure directories exist
LOGS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Trading schedule
MARKET_OPEN_TIME = "08:00:00"   # London Stock Exchange opening time (UTC)
MARKET_CLOSE_TIME = "16:30:00"  # London Stock Exchange closing time (UTC)
TRADING_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

# Logging settings
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Email notifications
ENABLE_EMAIL_NOTIFICATIONS = False
EMAIL_SENDER = "your_email@example.com"
EMAIL_PASSWORD = "your_app_password"  # Use app-specific password for Gmail
EMAIL_RECIPIENT = "your_email@example.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587 