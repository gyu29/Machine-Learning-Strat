#!/bin/bash

set -e

echo "Getting current S&P 500 buy/sell predictions..."
python3 predict.py

echo
echo "To get predictions again in the future, simply run:"
echo "./get_prediction.sh"
echo
echo "To retrain the models with the latest data, run:"
echo "./run_model_demo.sh"