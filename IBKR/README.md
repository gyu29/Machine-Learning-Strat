# IBKR Trading Bot

Trading bot for Interactive Brokers that trades Vanguard S&P 500 UCITS ETF (VUSA) using machine learning.

## Installation

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Install IBKR API from TWS installation directory:
```bash
cd ~/ibc/IBJts/source/pythonclient  # Adjust path as needed
python setup.py install
```

## Usage

1. Edit `config.py` to set connection settings and trading parameters.
2. Start TWS or IB Gateway with API connections enabled.
3. Run the bot:
```bash
./run_trading_bot.sh
```

Options:
- `--test`: Run a single execution cycle
- `--data-only`: Collect data without trading
- `--interval 1800`: Set check interval (seconds)

## Risk Warning

This software is for educational purposes only. Trading financial instruments involves risk of loss. 