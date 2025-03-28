import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.dates as mdates
from typing import Dict, List, Tuple, Union, Optional, Any

class MarketVisualizer:
    """Visualization tools for market data and predictions."""
    
    @staticmethod
    def plot_price_with_ma(df: pd.DataFrame, ma_periods: List[int] = [20, 50, 200], 
                         figsize: Tuple[int, int] = (12, 8)):
        """
        Plot price chart with moving averages.
        
        Args:
            df: DataFrame with market data
            ma_periods: List of moving average periods
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Plot price
        plt.plot(df.index, df['Close'], label='Price', color='black')
        
        # Plot moving averages
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        for i, period in enumerate(ma_periods):
            if f'SMA_{period}' in df.columns:
                plt.plot(df.index, df[f'SMA_{period}'], label=f'SMA {period}', color=colors[i % len(colors)])
        
        # Add labels and title
        plt.title('Price Chart with Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.gcf().autofmt_xdate()
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_technical_indicators(df: pd.DataFrame, figsize: Tuple[int, int] = (15, 15)):
        """
        Plot technical indicators.
        
        Args:
            df: DataFrame with technical indicators
            figsize: Figure size
        """
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        # Plot 1: Price and MAs
        axes[0].plot(df.index, df['Close'], label='Price', color='black')
        if 'SMA_20' in df.columns:
            axes[0].plot(df.index, df['SMA_20'], label='SMA 20', color='blue')
        if 'SMA_50' in df.columns:
            axes[0].plot(df.index, df['SMA_50'], label='SMA 50', color='green')
        if 'SMA_200' in df.columns:
            axes[0].plot(df.index, df['SMA_200'], label='SMA 200', color='red')
        
        # Add Bollinger Bands if available
        if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
            axes[0].fill_between(df.index, df['BB_Upper'], df['BB_Lower'], color='gray', alpha=0.2, label='Bollinger Bands')
        
        axes[0].set_title('Price with Moving Averages')
        axes[0].set_ylabel('Price')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: MACD
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns and 'MACD_Hist' in df.columns:
            axes[1].plot(df.index, df['MACD'], label='MACD', color='blue')
            axes[1].plot(df.index, df['MACD_Signal'], label='Signal', color='red')
            axes[1].bar(df.index, df['MACD_Hist'], label='Histogram', color='green', alpha=0.5)
            axes[1].axhline(y=0, color='black', linestyle='--')
            axes[1].set_title('MACD')
            axes[1].set_ylabel('Value')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # Plot 3: RSI
        if 'RSI_14' in df.columns:
            axes[2].plot(df.index, df['RSI_14'], label='RSI (14)', color='purple')
            axes[2].axhline(y=70, color='red', linestyle='--')
            axes[2].axhline(y=30, color='green', linestyle='--')
            axes[2].set_title('RSI')
            axes[2].set_ylabel('Value')
            axes[2].set_ylim(0, 100)
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Volume
        axes[3].bar(df.index, df['Volume'], label='Volume', color='blue', alpha=0.5)
        if 'Volume_SMA_20' in df.columns:
            axes[3].plot(df.index, df['Volume_SMA_20'], label='Volume SMA (20)', color='red')
        axes[3].set_title('Volume')
        axes[3].set_ylabel('Volume')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        # Format x-axis
        axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axes[3].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        fig.autofmt_xdate()
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_prediction_vs_actual(df: pd.DataFrame, prediction_col: str = 'Predicted_Value',
                                lower_bound_col: str = 'Prediction_Lower',
                                upper_bound_col: str = 'Prediction_Upper',
                                actual_col: str = 'Close',
                                figsize: Tuple[int, int] = (12, 8)):
        """
        Plot model predictions vs actual values.
        
        Args:
            df: DataFrame with predictions and actual values
            prediction_col: Column name for predictions
            lower_bound_col: Column name for lower confidence bound
            upper_bound_col: Column name for upper confidence bound
            actual_col: Column name for actual values
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Plot actual values
        plt.plot(df.index, df[actual_col], label='Actual', color='black')
        
        # Plot predictions
        if prediction_col in df.columns:
            plt.plot(df.index, df[prediction_col], label='Predicted', color='blue', alpha=0.7)
        
        # Plot confidence intervals
        if lower_bound_col in df.columns and upper_bound_col in df.columns:
            plt.fill_between(df.index, df[lower_bound_col], df[upper_bound_col], 
                           color='blue', alpha=0.2, label='Confidence Interval')
        
        # Add labels and title
        plt.title('Predicted vs Actual Values')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.gcf().autofmt_xdate()
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_trading_signals(df: pd.DataFrame, entry_col: str = 'Entry_Signal',
                          exit_col: str = 'Exit_Signal', price_col: str = 'Close',
                          figsize: Tuple[int, int] = (12, 8)):
        """
        Plot trading signals on price chart.
        
        Args:
            df: DataFrame with signals
            entry_col: Column name for entry signals
            exit_col: Column name for exit signals
            price_col: Column name for price
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Plot price
        plt.plot(df.index, df[price_col], label='Price', color='black')
        
        # Plot entry signals
        if entry_col in df.columns:
            entry_points = df[df[entry_col] == True]
            plt.scatter(entry_points.index, entry_points[price_col], marker='^', color='green', s=100, label='Entry Signal')
        
        # Plot exit signals
        if exit_col in df.columns:
            exit_points = df[df[exit_col] == True]
            plt.scatter(exit_points.index, exit_points[price_col], marker='v', color='red', s=100, label='Exit Signal')
        
        # Add labels and title
        plt.title('Trading Signals')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.gcf().autofmt_xdate()
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_interactive_price_chart(df: pd.DataFrame):
        """
        Create interactive price chart with Plotly.
        
        Args:
            df: DataFrame with market data
        """
        # Create figure with secondary y-axis
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                          vertical_spacing=0.02, 
                          row_heights=[0.7, 0.3])
        
        # Add price candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add volume bar chart
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color='rgba(0, 0, 255, 0.5)'
            ),
            row=2, col=1
        )
        
        # Add moving averages if available
        for period in [20, 50, 200]:
            col_name = f'SMA_{period}'
            if col_name in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[col_name],
                        name=f'SMA {period}',
                        line=dict(width=1.5)
                    ),
                    row=1, col=1
                )
        
        # Add Bollinger Bands if available
        if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['BB_Upper'],
                    name='Bollinger Upper',
                    line=dict(width=1, color='rgba(100, 100, 100, 0.3)')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['BB_Lower'],
                    name='Bollinger Lower',
                    fill='tonexty',
                    fillcolor='rgba(100, 100, 100, 0.1)',
                    line=dict(width=1, color='rgba(100, 100, 100, 0.3)')
                ),
                row=1, col=1
            )
        
        # Add entry/exit signals if available
        if 'Entry_Signal' in df.columns:
            entry_points = df[df['Entry_Signal'] == True]
            
            fig.add_trace(
                go.Scatter(
                    x=entry_points.index,
                    y=entry_points['Close'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=15,
                        color='green',
                        line=dict(width=2, color='DarkGreen')
                    ),
                    name='Entry Signal'
                ),
                row=1, col=1
            )
        
        if 'Exit_Signal' in df.columns:
            exit_points = df[df['Exit_Signal'] == True]
            
            fig.add_trace(
                go.Scatter(
                    x=exit_points.index,
                    y=exit_points['Close'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=15,
                        color='red',
                        line=dict(width=2, color='DarkRed')
                    ),
                    name='Exit Signal'
                ),
                row=1, col=1
            )
        
        # Update layout
        fig.update_layout(
            title='Interactive Price Chart',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            yaxis2_title='Volume',
            template='plotly_white',
            height=800,
            hovermode='x unified'
        )
        
        # Show figure
        fig.show()
    
    @staticmethod
    def plot_correlation_matrix(df: pd.DataFrame, columns: Optional[List[str]] = None,
                             figsize: Tuple[int, int] = (12, 10)):
        """
        Plot correlation matrix of selected columns.
        
        Args:
            df: DataFrame with data
            columns: List of columns to include (if None, use all numeric columns)
            figsize: Figure size
        """
        # Select columns to include
        if columns is None:
            # Use all numeric columns
            numeric_columns = df.select_dtypes(include=['number']).columns
            # Exclude date-related columns
            columns = [col for col in numeric_columns if not any(x in col.lower() for x in ['date', 'time', 'year', 'month', 'day'])]
        
        # Calculate correlation matrix
        corr_matrix = df[columns].corr()
        
        # Plot correlation matrix
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                  square=True, linewidths=0.5, annot=True, fmt='.2f', annot_kws={"size": 8})
        
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_feature_importance(feature_names: List[str], importances: np.ndarray, 
                             figsize: Tuple[int, int] = (12, 8), top_n: int = 20):
        """
        Plot feature importance.
        
        Args:
            feature_names: List of feature names
            importances: Array of feature importances
            figsize: Figure size
            top_n: Number of top features to show
        """
        # Sort features by importance
        indices = np.argsort(importances)[-top_n:]
        
        plt.figure(figsize=figsize)
        plt.title(f'Top {top_n} Feature Importance')
        plt.barh(range(top_n), importances[indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_macroeconomic_indicators(macro_data: Dict[str, pd.Series], 
                                   figsize: Tuple[int, int] = (15, 15)):
        """
        Plot macroeconomic indicators.
        
        Args:
            macro_data: Dictionary of macroeconomic indicator series
            figsize: Figure size
        """
        n_indicators = len(macro_data)
        fig, axes = plt.subplots(n_indicators, 1, figsize=figsize, sharex=True)
        
        # If there's only one indicator, make axes a list for consistent indexing
        if n_indicators == 1:
            axes = [axes]
        
        for i, (indicator_name, series) in enumerate(macro_data.items()):
            axes[i].plot(series.index, series.values)
            axes[i].set_title(indicator_name)
            axes[i].grid(True, alpha=0.3)
        
        # Format x-axis
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axes[-1].xaxis.set_major_locator(mdates.YearLocator())
        fig.autofmt_xdate()
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_returns_distribution(returns: pd.Series, bins: int = 50, 
                               figsize: Tuple[int, int] = (12, 8)):
        """
        Plot distribution of returns.
        
        Args:
            returns: Series of returns
            bins: Number of bins for histogram
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Plot histogram with KDE
        sns.histplot(returns, bins=bins, kde=True)
        
        # Add normal distribution fit
        x = np.linspace(returns.min(), returns.max(), 100)
        from scipy.stats import norm
        mu, sigma = norm.fit(returns)
        y = norm.pdf(x, mu, sigma)
        plt.plot(x, y, color='red', linestyle='--', label=f'Normal: μ={mu:.4f}, σ={sigma:.4f}')
        
        # Add vertical line at zero
        plt.axvline(x=0, color='black', linestyle='--')
        
        # Add mean and median lines
        plt.axvline(x=returns.mean(), color='green', linestyle='-', label=f'Mean: {returns.mean():.4f}')
        plt.axvline(x=returns.median(), color='blue', linestyle='-', label=f'Median: {returns.median():.4f}')
        
        # Add labels and title
        plt.title('Returns Distribution')
        plt.xlabel('Return')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print additional statistics
        print(f"Skewness: {returns.skew():.4f}")
        print(f"Kurtosis: {returns.kurtosis():.4f}")
        print(f"Value at Risk (95%): {np.percentile(returns, 5):.4f}")
        print(f"Expected Shortfall (95%): {returns[returns <= np.percentile(returns, 5)].mean():.4f}")
    
    @staticmethod
    def plot_drawdown_chart(equity_curve: pd.Series, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot drawdown chart.
        
        Args:
            equity_curve: Series of equity values
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Calculate drawdown
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve / rolling_max - 1) * 100
        
        # Plot drawdown
        plt.plot(drawdown, color='red')
        plt.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        
        # Add labels and title
        plt.title('Drawdown Chart')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        
        # Add horizontal lines at common drawdown levels
        plt.axhline(y=-5, color='gray', linestyle='--', alpha=0.5, label='-5%')
        plt.axhline(y=-10, color='gray', linestyle='--', alpha=0.5, label='-10%')
        plt.axhline(y=-20, color='gray', linestyle='--', alpha=0.5, label='-20%')
        
        # Format y-axis
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.gcf().autofmt_xdate()
        
        # Add max drawdown text
        max_drawdown = drawdown.min()
        max_drawdown_date = drawdown.idxmin()
        plt.annotate(f'Max Drawdown: {max_drawdown:.2f}%',
                   xy=(max_drawdown_date, max_drawdown),
                   xytext=(max_drawdown_date, max_drawdown - 5),
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                   fontsize=12)
        
        plt.tight_layout()
        plt.show() 