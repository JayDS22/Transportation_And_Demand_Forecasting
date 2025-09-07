import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import scipy.stats as stats

class ForecastingMetrics:
    """
    Comprehensive metrics for time series forecasting evaluation
    """
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive forecasting metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Percentage metrics
        metrics['mape'] = ForecastingMetrics.mean_absolute_percentage_error(y_true, y_pred)
        metrics['smape'] = ForecastingMetrics.symmetric_mean_absolute_percentage_error(y_true, y_pred)
        
        # Direction accuracy
        metrics['directional_accuracy'] = ForecastingMetrics.directional_accuracy(y_true, y_pred)
        
        # Statistical tests
        metrics['correlation'] = stats.pearsonr(y_true, y_pred)[0]
        metrics['correlation_pvalue'] = stats.pearsonr(y_true, y_pred)[1]
        
        # Residual analysis
        residuals = y_true - y_pred
        metrics['residual_mean'] = np.mean(residuals)
        metrics['residual_std'] = np.std(residuals)
        metrics['residual_skewness'] = stats.skew(residuals)
        metrics['residual_kurtosis'] = stats.kurtosis(residuals)
        
        # Theil's U statistic
        metrics['theil_u'] = ForecastingMetrics.theil_u_statistic(y_true, y_pred)
        
        return metrics
    
    @staticmethod
    def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate MAPE"""
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def symmetric_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate SMAPE"""
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
    
    @staticmethod
    def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate directional accuracy"""
        if len(y_true) < 2:
            return np.nan
        
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        
        return np.mean(true_direction == pred_direction) * 100
    
    @staticmethod
    def theil_u_statistic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Theil's U statistic"""
        mse = mean_squared_error(y_true, y_pred)
        
        # Naive forecast (random walk)
        naive_pred = np.concatenate([[y_true[0]], y_true[:-1]])
        naive_mse = mean_squared_error(y_true, naive_pred)
        
        if naive_mse == 0:
            return np.nan
        
        return np.sqrt(mse) / np.sqrt(naive_mse)
