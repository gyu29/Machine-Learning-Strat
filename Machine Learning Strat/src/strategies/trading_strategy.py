import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
class TradingStrategy:
    def __init__(self, name: str, prediction_model, 
                entry_threshold: float = 0.05, exit_threshold: float = 0.05,
                stop_loss: float = 0.1, take_profit: float = 0.2,
                max_position_duration: int = 120, min_position_duration: int = 20,
                max_drawdown: float = 0.15, position_size: float = 1.0,
                enable_risk_adjustment: bool = True):
        self.name = name
        self.prediction_model = prediction_model
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_position_duration = max_position_duration
        self.min_position_duration = min_position_duration
        self.max_drawdown = max_drawdown
        self.position_size = position_size
        self.enable_risk_adjustment = enable_risk_adjustment
        self.in_position = False
        self.position_entry_price = None
        self.position_entry_date = None
        self.position_size_current = position_size
        self.equity_curve = []
        self.trades = []
        self.current_drawdown = 0
        self.max_drawdown_experienced = 0
        self.model_type = prediction_model.model_type
        self.target_horizon = prediction_model.target_horizon
        self.metadata = {
            'name': name,
            'model_name': prediction_model.model_name,
            'model_type': prediction_model.model_type,
            'target_horizon': prediction_model.target_horizon,
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'max_position_duration': max_position_duration,
            'min_position_duration': min_position_duration,
            'max_drawdown': max_drawdown,
            'position_size': position_size,
            'enable_risk_adjustment': enable_risk_adjustment
        }
    def generate_signals(self, df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        result = df.copy()
        X = result[feature_columns].values
        predictions = self.prediction_model.predict(X)
        lower_bound, upper_bound = self.prediction_model.get_confidence_interval(X)
        if self.model_type in ['price', 'return']:
            result['Predicted_Value'] = predictions
            result['Prediction_Lower'] = lower_bound
            result['Prediction_Upper'] = upper_bound
            if self.model_type == 'return':
                result['Expected_Return'] = predictions
                result['Entry_Signal'] = (predictions > self.entry_threshold) & (lower_bound > 0)
                result['Exit_Signal'] = (predictions < self.exit_threshold) | (upper_bound < 0)
            else:
                result['Expected_Return'] = (predictions / result['Close'] - 1)
                result['Entry_Signal'] = (result['Expected_Return'] > self.entry_threshold) & (lower_bound > result['Close'])
                result['Exit_Signal'] = (result['Expected_Return'] < self.exit_threshold) | (upper_bound < result['Close'])
        else:
            result['Predicted_Direction'] = predictions
            result['Direction_Prob'] = (upper_bound + 1 - lower_bound) / 2
            result['Entry_Signal'] = (predictions == 1) & (result['Direction_Prob'] > (0.5 + self.entry_threshold))
            result['Exit_Signal'] = (predictions == 0) | (result['Direction_Prob'] < (0.5 + self.exit_threshold))
        return result
    def backtest(self, df: pd.DataFrame, feature_columns: List[str], 
                initial_capital: float = 100000.0, trading_fee: float = 0.001) -> Dict[str, Any]:
        strategy_df = self.generate_signals(df, feature_columns)
        self.in_position = False
        self.position_entry_price = None
        self.position_entry_date = None
        self.position_size_current = self.position_size
        capital = initial_capital
        self.equity_curve = [initial_capital]
        self.trades = []
        self.current_drawdown = 0
        self.max_drawdown_experienced = 0
        slippage = 0.001
        for i in range(1, len(strategy_df)):
            current_date = strategy_df.index[i]
            current_price = strategy_df.iloc[i]['Close']
            prev_price = strategy_df.iloc[i-1]['Close']
            if self.in_position:
                unrealized_return = (current_price / self.position_entry_price - 1)
                new_capital = capital * (1 + self.position_size_current * unrealized_return)
                self.equity_curve.append(new_capital)
                peak_capital = max(self.equity_curve)
                self.current_drawdown = (peak_capital - new_capital) / peak_capital
                self.max_drawdown_experienced = max(self.max_drawdown_experienced, self.current_drawdown)
                position_duration = (current_date - self.position_entry_date).days
                exit_signal = strategy_df.iloc[i]['Exit_Signal']
                stop_loss_hit = unrealized_return < -self.stop_loss
                take_profit_hit = unrealized_return > self.take_profit
                max_duration_hit = position_duration >= self.max_position_duration
                max_drawdown_hit = self.current_drawdown >= self.max_drawdown
                min_duration_passed = position_duration >= self.min_position_duration
                if min_duration_passed and (exit_signal or stop_loss_hit or take_profit_hit or max_duration_hit):
                    realized_return = (current_price / self.position_entry_price - 1) - trading_fee - slippage
                    capital = capital * (1 + self.position_size_current * realized_return)
                    self.in_position = False
                    self.trades.append({
                        'entry_date': self.position_entry_date,
                        'exit_date': current_date,
                        'entry_price': self.position_entry_price,
                        'exit_price': current_price,
                        'position_size': self.position_size_current,
                        'return': realized_return,
                        'pnl': capital * realized_return,
                        'duration': position_duration,
                        'exit_reason': 'signal' if exit_signal else 'stop_loss' if stop_loss_hit else 'take_profit' if take_profit_hit else 'max_duration'
                    })
                    if max_drawdown_hit:
                        self.current_drawdown = 0
                elif max_drawdown_hit:
                    realized_return = (current_price / self.position_entry_price - 1) - trading_fee - slippage
                    capital = capital * (1 + self.position_size_current * realized_return)
                    self.in_position = False
                    self.trades.append({
                        'entry_date': self.position_entry_date,
                        'exit_date': current_date,
                        'entry_price': self.position_entry_price,
                        'exit_price': current_price,
                        'position_size': self.position_size_current,
                        'return': realized_return,
                        'pnl': capital * realized_return,
                        'duration': position_duration,
                        'exit_reason': 'max_drawdown'
                    })
                    self.current_drawdown = 0
            else:
                self.equity_curve.append(capital)
                entry_signal = strategy_df.iloc[i]['Entry_Signal']
                if self.enable_risk_adjustment:
                    if i >= 20:
                        recent_returns = strategy_df.iloc[i-20:i]['Close'].pct_change().dropna()
                        volatility = recent_returns.std() * np.sqrt(252)
                        target_vol = 0.15
                        self.position_size_current = self.position_size * (target_vol / volatility)
                        self.position_size_current = min(self.position_size_current, self.position_size * 2)
                        self.position_size_current = max(self.position_size_current, self.position_size * 0.5)
                if entry_signal and self.current_drawdown < self.max_drawdown:
                    self.in_position = True
                    self.position_entry_price = current_price
                    self.position_entry_date = current_date
        equity_series = pd.Series(self.equity_curve, index=strategy_df.index[:len(self.equity_curve)])
        returns = equity_series.pct_change().dropna()
        annual_return = (equity_series.iloc[-1] / equity_series.iloc[0]) ** (252 / len(equity_series)) - 1
        annual_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        max_drawdown = self.max_drawdown_experienced
        if len(self.trades) > 0:
            trades_df = pd.DataFrame(self.trades)
            win_rate = (trades_df['return'] > 0).mean()
            avg_win = trades_df.loc[trades_df['return'] > 0, 'return'].mean() if any(trades_df['return'] > 0) else 0
            avg_loss = trades_df.loc[trades_df['return'] < 0, 'return'].mean() if any(trades_df['return'] < 0) else 0
            profit_factor = -avg_win / avg_loss if avg_loss < 0 else float('inf')
            avg_duration = trades_df['duration'].mean()
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            avg_duration = 0
        results = {
            'equity_curve': equity_series,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_duration': avg_duration,
            'final_capital': self.equity_curve[-1],
            'total_return': (self.equity_curve[-1] / self.equity_curve[0]) - 1,
            'trades': self.trades
        }
        return results
    def plot_equity_curve(self, results: Dict[str, Any], benchmark_returns: Optional[pd.Series] = None):
        plt.figure(figsize=(12, 8))
        equity_curve = results['equity_curve']
        plt.plot(equity_curve, label=f'Strategy: {self.name}')
        if benchmark_returns is not None:
            benchmark_equity = (1 + benchmark_returns).cumprod() * equity_curve.iloc[0]
            plt.plot(benchmark_equity, label='Benchmark (S&P 500)', alpha=0.7)
        drawdowns = equity_curve / equity_curve.cummax() - 1
        plt.fill_between(equity_curve.index, 0, drawdowns * 100, color='red', alpha=0.3, label='Drawdown %')
        plt.title(f'Equity Curve - {self.name}')
        plt.xlabel('Date')
        plt.ylabel('Capital ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        textstr = '\n'.join((
            f'Annual Return: {results["annual_return"]:.2%}',
            f'Sharpe Ratio: {results["sharpe_ratio"]:.2f}',
            f'Max Drawdown: {results["max_drawdown"]:.2%}',
            f'Win Rate: {results["win_rate"]:.2%}',
            f'Total Trades: {results["total_trades"]}'
        ))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.annotate(textstr, xy=(0.05, 0.95), xycoords='axes fraction', 
                    verticalalignment='top', bbox=props)
        plt.tight_layout()
        plt.show()
    def plot_trade_distribution(self, results: Dict[str, Any]):
        if len(results['trades']) == 0:
            print("No trades to plot")
            return
        trades_df = pd.DataFrame(results['trades'])
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        sns.histplot(trades_df['return'], kde=True)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('Trade Return Distribution')
        plt.xlabel('Return')
        plt.subplot(2, 2, 2)
        plt.scatter(trades_df['duration'], trades_df['return'], alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Return vs Duration')
        plt.xlabel('Duration (days)')
        plt.ylabel('Return')
        plt.subplot(2, 2, 3)
        cumulative_returns = (1 + trades_df['return']).cumprod() - 1
        plt.plot(cumulative_returns)
        plt.title('Cumulative Trade Returns')
        plt.xlabel('Trade #')
        plt.ylabel('Cumulative Return')
        plt.subplot(2, 2, 4)
        trades_df['yearmonth'] = trades_df['exit_date'].dt.to_period('M')
        monthly_returns = trades_df.groupby('yearmonth')['return'].sum()
        monthly_returns.plot(kind='bar')
        plt.title('Monthly Returns')
        plt.xlabel('Month')
        plt.ylabel('Return')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    def save_results(self, results: Dict[str, Any], path: str):
        import os
        import json
        import pickle
        if not os.path.exists(path):
            os.makedirs(path)
        results['equity_curve'].to_csv(os.path.join(path, f"{self.name}_equity_curve.csv"))
        if len(results['trades']) > 0:
            pd.DataFrame(results['trades']).to_csv(os.path.join(path, f"{self.name}_trades.csv"))
        metrics = {k: v for k, v in results.items() if k not in ['equity_curve', 'trades']}
        metrics['strategy'] = self.metadata
        with open(os.path.join(path, f"{self.name}_metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=4, default=str)
        with open(os.path.join(path, f"{self.name}_results.pkl"), 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to {path}")
    def optimize_parameters(self, df: pd.DataFrame, feature_columns: List[str],
                          param_grid: Dict[str, List[Any]], initial_capital: float = 100000.0):
        import itertools
        param_names = list(param_grid.keys())
        param_values = list(itertools.product(*param_grid.values()))
        best_sharpe = -float('inf')
        best_params = None
        best_results = None
        for i, values in enumerate(param_values):
            params = dict(zip(param_names, values))
            for param_name, param_value in params.items():
                setattr(self, param_name, param_value)
            for param_name, param_value in params.items():
                self.metadata[param_name] = param_value
            results = self.backtest(df, feature_columns, initial_capital)
            if results['sharpe_ratio'] > best_sharpe:
                best_sharpe = results['sharpe_ratio']
                best_params = params.copy()
                best_results = results.copy()
            print(f"Tested parameters {i+1}/{len(param_values)}: {params}")
            print(f"Sharpe: {results['sharpe_ratio']:.2f}, Return: {results['annual_return']:.2%}")
        for param_name, param_value in best_params.items():
            setattr(self, param_name, param_value)
            self.metadata[param_name] = param_value
        return {
            'best_params': best_params,
            'best_sharpe': best_sharpe,
            'best_results': best_results
        }