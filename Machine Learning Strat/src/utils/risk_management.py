import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
class RiskManager:
    @staticmethod
    def calculate_position_size(capital: float, risk_per_trade: float, stop_loss: float) -> float:
        dollar_risk = capital * risk_per_trade
        position_size = dollar_risk / (capital * stop_loss)
        position_size = min(position_size, 1.0)
        return position_size
    @staticmethod
    def adjust_position_size_by_volatility(position_size: float, current_volatility: float, 
                                         target_volatility: float, max_adjustment: float = 2.0) -> float:
        adjustment = target_volatility / current_volatility
        adjustment = min(adjustment, max_adjustment)
        adjustment = max(adjustment, 1.0 / max_adjustment)
        adjusted_position_size = position_size * adjustment
        adjusted_position_size = min(adjusted_position_size, position_size * max_adjustment)
        adjusted_position_size = min(adjusted_position_size, 1.0)
        return adjusted_position_size
    @staticmethod
    def calculate_volatility(returns: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
        volatility = returns.rolling(window=window).std()
        if annualize:
            volatility = volatility * np.sqrt(252)
        return volatility
    @staticmethod
    def calculate_value_at_risk(returns: pd.Series, confidence_level: float = 0.95, 
                              window: int = 252) -> pd.Series:
        alpha = 1 - confidence_level
        var = returns.rolling(window=window).quantile(alpha)
        return var
    @staticmethod
    def calculate_expected_shortfall(returns: pd.Series, confidence_level: float = 0.95, 
                                   window: int = 252) -> pd.Series:
        alpha = 1 - confidence_level
        def expected_shortfall(x, alpha):
            sorted_returns = np.sort(x)
            cutoff_index = int(np.ceil(alpha * len(x))) - 1
            if cutoff_index >= 0:
                return sorted_returns[:cutoff_index+1].mean()
            else:
                return sorted_returns[0]
        es = returns.rolling(window=window).apply(lambda x: expected_shortfall(x, alpha), raw=True)
        return es
    @staticmethod
    def calculate_maximum_drawdown(prices: pd.Series) -> float:
        running_max = prices.cummax()
        drawdown = (prices / running_max - 1)
        max_drawdown = drawdown.min()
        return max_drawdown
    @staticmethod
    def calculate_risk_metrics(returns: pd.Series) -> Dict[str, float]:
        annual_return = ((1 + returns).prod()) ** (252 / len(returns)) - 1
        annual_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annual_return / downside_volatility if downside_volatility > 0 else 0
        cumulative_returns = (1 + returns).cumprod()
        max_drawdown = RiskManager.calculate_maximum_drawdown(cumulative_returns)
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
        var_95 = np.percentile(returns, 5)
        es_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'var_95': var_95,
            'expected_shortfall_95': es_95,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
    @staticmethod
    def calculate_rolling_risk_metrics(returns: pd.Series, window: int = 252) -> pd.DataFrame:
        metrics = pd.DataFrame(index=returns.index)
        metrics['volatility'] = RiskManager.calculate_volatility(returns, window)
        metrics['var_95'] = RiskManager.calculate_value_at_risk(returns, 0.95, window)
        metrics['es_95'] = RiskManager.calculate_expected_shortfall(returns, 0.95, window)
        rolling_returns = returns.rolling(window=window)
        annual_return = (rolling_returns.apply(lambda x: (1 + x).prod()) ** (252 / window) - 1)
        metrics['sharpe_ratio'] = annual_return / metrics['volatility']
        def rolling_max_drawdown(x):
            cumulative_returns = (1 + x).cumprod()
            return RiskManager.calculate_maximum_drawdown(cumulative_returns)
        metrics['max_drawdown'] = returns.rolling(window=window).apply(rolling_max_drawdown)
        return metrics
    @staticmethod
    def calculate_risk_contribution(weights: np.ndarray, cov_matrix: pd.DataFrame) -> np.ndarray:
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        marginal_contrib = np.dot(cov_matrix, weights)
        risk_contrib = np.multiply(marginal_contrib, weights) / portfolio_vol
        return risk_contrib
    @staticmethod
    def equal_risk_contribution(cov_matrix: pd.DataFrame, max_iterations: int = 100, 
                              tolerance: float = 1e-8) -> np.ndarray:
        n = cov_matrix.shape[0]
        weights = np.ones(n) / n
        for iter in range(max_iterations):
            risk_contrib = RiskManager.calculate_risk_contribution(weights, cov_matrix)
            target_risk = 1.0 / n
            adjustment = target_risk / risk_contrib
            new_weights = weights * adjustment
            new_weights = new_weights / np.sum(new_weights)
            if np.sqrt(np.sum((new_weights - weights) ** 2)) < tolerance:
                weights = new_weights
                break
            weights = new_weights
        return weights
