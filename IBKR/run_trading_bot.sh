#!/bin/bash

cd "$(dirname "$0")"

mkdir -p logs
mkdir -p data

python main.py "$@" 