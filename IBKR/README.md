# IBKR Trading Bot for Vanguard S&P 500 UCITS ETF

This module implements an automated trading bot for Interactive Brokers that trades the Vanguard S&P 500 UCITS ETF (VUSA) based on machine learning predictions.

## Features

- Real-time trading with IBKR API
- Machine learning-based signal generation
- Risk management with stop-loss, take-profit, and position sizing
- Email notifications for trade events
- Automatic market hours detection
- Data collection and enrichment

## Prerequisites

1. An Interactive Brokers account
2. TWS (Trader Workstation) or IB Gateway installed and running
3. API access enabled in TWS/IB Gateway (File → Global Configuration → API → Enable ActiveX and Socket Clients)
4. Python 3.7+ with required packages
5. Trained machine learning models

## Installation

1. Install the required Python packages:

```bash
pip install ibapi pandas numpy matplotlib scikit-learn
```

2. The IBKR API (ibapi) package is not available on PyPI. It must be installed from the TWS installation:

```bash
# For Mac/Linux (adjust path as needed)
cd ~/ibc/IBJts/source/pythonclient
python setup.py install

# For Windows (adjust path as needed)
cd C:\Jts\source\pythonclient
python setup.py install
```

## Configuration

Edit the `config.py` file to customize:

- IBKR connection settings
- ETF details
- Trading parameters (entry/exit thresholds, stop-loss, etc.)
- Email notifications

## Usage

### Start TWS or IB Gateway

Before running the bot, make sure TWS or IB Gateway is running and API connections are enabled.

### Basic Usage

Run the bot in standard mode:

```bash
python main.py
```

### Test Mode

Run a single execution cycle without continuous trading:

```bash
python main.py --test
```

### Data Collection Only

Only collect market data without trading:

```bash
python main.py --data-only
```

### Custom Check Interval

Set a custom interval for trade signal checking (in seconds):

```bash
python main.py --interval 1800  # Check every 30 minutes
```

## Trading Logic

The bot follows these steps:

1. Connects to IBKR API and loads ML models
2. Retrieves market data for VUSA and enriches it with technical indicators
3. Generates trading signals using the loaded ML model
4. If not in a position and a BUY signal is generated:
   - Executes a market order to enter the position
   - Sets a stop-loss order
5. If in a position, checks exit criteria:
   - Stop-loss hit
   - Take-profit target reached
   - Maximum position duration exceeded
   - ML model generates SELL signal
6. Repeats the process at specified intervals during market hours

## Risk Management

- Position sizing: Allocates a percentage of account value to each trade
- Stop-loss: Automatic stop orders to limit losses
- Take-profit: Exits positions when profit targets are reached
- Duration limits: Maximum holding period to prevent indefinite positions
- Drawdown protection: Exits when portfolio drawdown exceeds thresholds

## Logs and Monitoring

Logs are stored in the `logs` directory with detailed information on:
- Trading signals and decisions
- Order execution
- Position status
- Errors and warnings

## Disclaimer

This software is for educational and research purposes only. Use at your own risk. Trading financial instruments involves risk of loss.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 