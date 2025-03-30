#!/bin/bash

# Change to the IBKR directory
cd "$(dirname "$0")"

# Ensure logs directory exists
mkdir -p logs
mkdir -p data

# Run the trading bot with command line arguments
python main.py "$@" 