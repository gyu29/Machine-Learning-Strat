# Long-Term S&P 500 Price Prediction and Trading Strategy

A comprehensive machine learning framework for predicting S&P 500 price movements and implementing a data-driven trading strategy across multiple time horizons.

## Features

- **Multi-dimensional prediction model** incorporating macroeconomic, technical, and fundamental analysis
- **Advanced technical analysis** including Fibonacci retracement levels, moving average convergence, and trend channel analysis
- **Multiple prediction horizons** (1-3 month primary forecast, 6-12 month strategic outlook)
- **Ensemble machine learning models** including Random Forest, LSTM networks, and Gradient Boosting algorithms
- **Comprehensive risk management framework** with adaptive stop-loss mechanisms and volatility-based position sizing
- **Backtesting system** with detailed performance metrics and visualization tools
- **Continuous model improvement** methodology with quarterly model recalibration

## Project Structure

```
├── src                     # Source code
│   ├── data                # Data acquisition and processing
│   ├── models              # Prediction models 
│   ├── strategies          # Trading strategy implementation
│   ├── utils               # Utility functions
│   └── visualization       # Data visualization tools
├── notebooks               # Jupyter notebooks for analysis
├── tests                   # Unit tests
├── README.md               # Project documentation
└── requirements.txt        # Project dependencies
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/gyu29/Machine-Learning-Strat
   cd sp500-prediction-strategy
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Note: If you encounter issues installing TA-Lib, please follow the specific instructions for your operating system available at [TA-Lib Installation Guide](https://github.com/TA-Lib/ta-lib-python).

## Getting Started

1. **Data Collection**
   - You'll need a FRED API key for macroeconomic data (available at https://fred.stlouisfed.org/docs/api/api_key.html)
   - Open the example notebook and replace `'YOUR_FRED_API_KEY'` with your actual API key

2. **Run the Example Notebook**
   ```bash
   jupyter notebook notebooks/example_workflow.ipynb
   ```

   This notebook demonstrates:
   - Data collection from various sources
   - Feature engineering 
   - Model training and evaluation
   - Trading strategy implementation
   - Backtesting and performance analysis
   - Strategy optimization

   ***NOTE***  
   The jupiter notebook is not finished yet

## S&P 500 Buy/Sell Predictions

### Quick Start

Follow these steps to get buy/sell predictions:

1. First, train the models with current data:
   ```bash
   ./run_model_demo.sh
   ```

2. Then, get the current prediction:
   ```bash
   python predict.py
   ```

That's it! The prediction script will display a BUY or SELL recommendation for the S&P 500 over the next 20 trading days (approximately 1 month), along with a confidence level.

### Understanding the Prediction

The prediction includes:
- **Current S&P 500 Price**: The most recent closing price
- **Predicted Price**: The expected price in 20 trading days
- **Predicted Return**: The expected percentage return
- **Direction Signal**: BUY or SELL recommendation
- **Signal Confidence**: How confident the model is in its prediction
- **Recommendation Strength**: STRONG, MODERATE, or WEAK based on confidence level

### How It Works

1. The project uses your FRED API key (stored in `src/config.py`) to fetch:
   - S&P 500 price data
   - Macroeconomic indicators (GDP, unemployment, interest rates, etc.)

2. It trains two machine learning models:
   - A price prediction model (predicts the actual price)
   - A direction prediction model (predicts whether the market will go up or down)

3. These models analyze patterns in the data to identify buy/sell signals.

### Updating Predictions

- Run `python predict.py` any time you want an updated prediction
- Run `./run_model_demo.sh` periodically (e.g., monthly) to retrain the models with the latest data

## Key Components

### Data Loaders
- `MarketDataLoader`: Fetches market data, macroeconomic indicators, and sector ETFs
- `FeatureEngineer`: Generates technical indicators and prepares features for modeling

### Models
- `BaseModel`: Abstract base class for all prediction models
- `RandomForestModel`: Implementation using Random Forest algorithm
- `LSTMModel`: Implementation using LSTM neural networks
- `EnsembleModel`: Combines multiple models for improved predictions

### Trading Strategy
- `TradingStrategy`: Implements the trading strategy based on model predictions
- `RiskManager`: Provides risk management utilities

### Visualization
- `MarketVisualizer`: Tools for visualizing market data, predictions, and performance metrics

## Prediction Methodology

1. **Multi-dimensional prediction model**
   - Macroeconomic indicator analysis (GDP, inflation, Fed policy, unemployment)
   - Technical price pattern recognition
   - Fundamental market health assessment

2. **Advanced Technical Analysis**
   - Fibonacci retracement levels
   - Moving average convergence
   - Long-term trend channel analysis
   - Support and resistance zone identification

3. **Ensemble Prediction Methodology**
   - Model consensus approach
   - Weighted prediction integration
   - Confidence interval calculation

## Trading Strategy

1. **Position Management**
   - Minimum 1-month position duration
   - Maximum 6-month position length
   - Quarterly strategic rebalancing
   - Risk-adjusted allocation

2. **Entry and Exit Criteria**
   - Confirmed long-term uptrend
   - Macroeconomic opportunity window
   - Technical support level validation
   - Trend reversal signals
   - Profit target achievement

3. **Risk Management**
   - Adaptive stop-loss mechanisms
   - Volatility-based position sizing
   - Maximum drawdown protection
   - Systematic position reduction

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. It is not intended to be trading or investment advice. Always do your own research and consult with a professional financial advisor before making investment decisions. 