set -e

echo "Running S&P 500 Price Prediction and Trading Strategy..."
echo "This will generate models that predict buy/sell signals for the S&P 500"
echo "using macroeconomic data from FRED with your API key"
echo

python3 src/main.py 

echo 
echo "Model training and prediction complete!"
echo "The trained models are saved in the 'models/' directory"
echo
echo "To view the visualizations, check the PNG files in the project directory:"
echo "- price_predictions.png - Shows model predictions vs actual prices"
echo "- price_feature_importance.png - Shows which features are most important"
echo "- correlation_heatmap.png - Shows correlations between features"
echo
echo "The direction model predicts whether the market will go up (buy) or down (sell)"
echo "over a 20-day horizon based on current market and economic conditions." 