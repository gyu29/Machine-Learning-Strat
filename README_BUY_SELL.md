# S&P 500 Buy/Sell Prediction Guide

This document explains how to use this project to generate buy/sell signals for the S&P 500.

## Quick Start

Follow these steps to get buy/sell predictions:

1. First, train the models with current data:
   ```bash
   ./run_model_demo.sh
   ```

2. Then, get the current prediction:
   ```bash
   ./get_prediction.sh
   ```

That's it! The prediction script will display a BUY or SELL recommendation for the S&P 500 over the next 20 trading days (approximately 1 month), along with a confidence level.

## Understanding the Prediction

The prediction includes:
- **Current S&P 500 Price**: The most recent closing price
- **Predicted Price**: The expected price in 20 trading days
- **Predicted Return**: The expected percentage return
- **Direction Signal**: BUY or SELL recommendation
- **Signal Confidence**: How confident the model is in its prediction
- **Recommendation Strength**: STRONG, MODERATE, or WEAK based on confidence level

## How It Works

1. The project uses your FRED API key (stored in `src/config.py`) to fetch:
   - S&P 500 price data
   - Macroeconomic indicators (GDP, unemployment, interest rates, etc.)

2. It trains two machine learning models:
   - A price prediction model (predicts the actual price)
   - A direction prediction model (predicts whether the market will go up or down)

3. These models analyze patterns in the data to identify buy/sell signals.

## Updating Predictions

- Run `./get_prediction.sh` any time you want an updated prediction
- Run `./run_model_demo.sh` periodically (e.g., monthly) to retrain the models with the latest data

## Disclaimer

This software is for educational and research purposes only. It is not intended to be trading or investment advice. Always do your own research and consult with a professional financial advisor before making investment decisions. 