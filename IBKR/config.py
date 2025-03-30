import os
from pathlib import Path

TWS_HOST = "127.0.0.1"
TWS_PORT = 7497
CLIENT_ID = 1

SYMBOL = "VUSA"
EXCHANGE = "LSE"
CURRENCY = "GBP"
SECURITY_TYPE = "STK"

ENTRY_THRESHOLD = 0.05
EXIT_THRESHOLD = 0.05
STOP_LOSS = 0.10
TAKE_PROFIT = 0.20
MAX_POSITION_DURATION = 60
MIN_POSITION_DURATION = 20
POSITION_SIZE = 0.10
MAX_DRAWDOWN = 0.15
ENABLE_RISK_ADJUSTMENT = True

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = BASE_DIR.parent / "models"
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"

LOGS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

MARKET_OPEN_TIME = "08:00:00"
MARKET_CLOSE_TIME = "16:30:00"
TRADING_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

LOG_LEVEL = "INFO"

ENABLE_EMAIL_NOTIFICATIONS = False
EMAIL_SENDER = "your_email@example.com"
EMAIL_PASSWORD = "your_app_password"
EMAIL_RECIPIENT = "your_email@example.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587