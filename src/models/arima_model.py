import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
from typing import Tuple, Dict, Any, Optional
import logging

class ARIMADemandPredictor:
    """
    ARIMA model for transportation demand prediction with statistical tests
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.fitted_model = None
        self.logger = logging.getLogger(__name__)
        self.decomposition = None
        self.stationarity_results = {}
        
    def check_stationarity(self, series: pd.Series) -> Dict[str, Any]:
        """
        Comprehensive stationarity testing
        """
        results = {}
        
        # Augmented Dickey-Fuller test
        adf_result = adfuller(series.dropna())
        results['adf'] = {
            'statistic': adf_result[0],
            'p_value': adf_result[1],
            'critical_values': adf_result[4],
            'is_stationary': adf_result[1] < 0.05
        }
        
        # KPSS test
        kpss_result = kpss(series.dropna())
        results['kpss'] = {
            'statistic': kpss_result[0],
            'p_value': kpss_result[1],
            'critical_values': kpss_result[3],
            'is_stationary': kpss_result[1] > 0.05
        }
        
        # Combined interpretation
        adf_stationary = results['adf']['is_stationary']
        kpss_stationary = results['kpss']['is_stationary']
        
        if adf_stationary and kpss_stationary:
            results['conclusion'] = 'Stationary'
        elif not adf_stationary and not kpss_stationary:
            results['conclusion'] = 'Non-stationary'
        else:
            results['conclusion'] = 'Inconclusive - requires further investigation'
        
        self.stationarity_results = results
        self.logger.info(f"Stationarity test conclusion: {results['conclusion']}")
        
        return results
    
    def seasonal_decomposition(self, series: pd.Series, 
                             period: int = 24) -> Dict[str, pd.Series]:
        """
        Perform seasonal decomposition
        """
        self.decomposition = seasonal_decompose(
            series.dropna(), 
            model='additive', 
            period=period
        )
        
        components = {
            'trend': self.decomposition.trend,
            'seasonal': self.decomposition.seasonal,
            'residual': self.decomposition.resid,
            'observed': self.decomposition.observed
        }
        
        # Calculate seasonal strength
        seasonal_strength = 1 - (
            np.var(components['residual'].dropna()) / 
            np.var(components['seasonal'].dropna() + components['residual'].dropna())
        )
        
        self.logger.info(f"Seasonal strength: {seasonal_strength:.4f}")
        
        return components
    
    def auto_arima_selection(self, series: pd.Series, 
                           max_p: int = 5, max_d: int = 2, 
                           max_q: int = 5) -> Tuple[int, int, int]:
        """
        Automatic ARIMA order selection using AIC/BIC
        """
        best_aic = float('inf')
        best_bic = float('inf')
        best_order_aic = None
        best_order_bic = None
        
        results = []
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore")
                            model = ARIMA(series, order=(p, d, q))
                            fitted_model = model.fit()
                            
                            aic = fitted_model.aic
                            bic = fitted_model.bic
                            
                            results.append({
                                'order': (p, d, q),
                                'aic': aic,
                                'bic': bic
                            })
                            
                            if aic < best_aic:
                                best_aic = aic
                                best_order_aic = (p, d, q)
                            
                            if bic < best_bic:
                                best_bic = bic
                                best_order_bic = (p, d, q)
                                
                    except:
                        continue
        
        # Use AIC for selection (can be configured)
        criterion = self.config.get('selection_criterion', 'aic')
        if criterion == 'aic':
            best_order = best_order_aic
            best_score = best_aic
        else:
            best_order = best_order_bic
            best_score = best_bic
        
        self.logger.info(
            f"Best ARIMA order: {best_order} with {criterion.upper()}: {best_score:.2f}"
        )
        
        return best_order
    
    def fit(self, series: pd.Series, exog: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Fit ARIMA model with comprehensive diagnostics
        """
        # Check stationarity
        self.check_stationarity(series)
        
        # Seasonal decomposition
        self.seasonal_decomposition(series)
        
        # Auto-select ARIMA order
        if self.config.get('auto_arima', True):
            order = self.auto_arima_selection(series)
        else:
            order = self.config.get('arima_order', (1, 1, 1))
        
        # Fit the model
        try:
            self.model = ARIMA(series, order=order, exog=exog)
            self.fitted_model = self.model.fit()
            
            # Model diagnostics
            diagnostics = self._model_diagnostics()
            
            fit_results = {
                'order': order,
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic,
                'log_likelihood': self.fitted_model.llf,
                'diagnostics': diagnostics,
                'summary': str(self.fitted_model.summary())
            }
            
            self.logger.info(f"ARIMA{order} fitted successfully")
            return fit_results
            
        except Exception as e:
            self.logger.error(f"Error fitting ARIMA model: {e}")
            raise
    
    def _model_diagnostics(self) -> Dict[str, Any]:
        """
        Comprehensive model diagnostics
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted yet")
        
        residuals = self.fitted_model.resid
        
        # Ljung-Box test for autocorrelation in residuals
        lb_test = acorr_ljungbox(residuals, lags=min(10, len(residuals)//5))
        
        # Normality test (Jarque-Bera is included in model summary)
        jb_stat = self.fitted_model.jarque_bera
        
        diagnostics = {
            'ljung_box': {
                'statistic': lb_test['lb_stat'].iloc[-1],
                'p_value': lb_test['lb_pvalue'].iloc[-1],
                'no_autocorrelation': lb_test['lb_pvalue'].iloc[-1] > 0.05
            },
            'jarque_bera': {
                'statistic': jb_stat[0],
                'p_value': jb_stat[1],
                'normal_residuals': jb_stat[1] > 0.05
            },
            'residual_stats': {
                'mean': float(residuals.mean()),
                'std': float(residuals.std()),
                'skewness': float(residuals.skew()),
                'kurtosis': float(residuals.kurtosis())
            }
        }
        
        return diagnostics
    
    def granger_causality_test(self, y: pd.Series, x: pd.DataFrame, 
                              max_lags: int = 4) -> Dict[str, Any]:
        """
        Test Granger causality between external variables and demand
        """
        causality_results = {}
        
        for column in x.columns:
            try:
                # Combine series for testing
                data = pd.concat([y, x[column]], axis=1).dropna()
                
                # Test with multiple lags
                test_result = grangercausalitytests(
                    data, 
                    maxlag=max_lags, 
                    verbose=False
                )
                
                # Extract p-values for each lag
                p_values = []
                for lag in range(1, max_lags + 1):
                    if lag in test_result:
                        p_value = test_result[lag][0]['ssr_ftest'][1]
                        p_values.append(p_value)
                
                min_p_value = min(p_values) if p_values else 1.0
                
                causality_results[column] = {
                    'min_p_value': min_p_value,
                    'is_causal': min_p_value < 0.05,
                    'optimal_lag': p_values.index(min_p_value) + 1 if p_values else None
                }
                
            except Exception as e:
                self.logger.warning(f"Granger causality test failed for {column}: {e}")
                causality_results[column] = {'error': str(e)}
        
        self.logger.info("Granger causality testing completed")
        return causality_results
    
    def predict(self, steps: int, exog: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Make predictions with confidence intervals
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted yet")
        
        try:
            forecast_result = self.fitted_model.forecast(
                steps=steps, 
                exog=exog
            )
            
            # Get prediction intervals
            pred_ci = self.fitted_model.get_prediction(
                start=len(self.fitted_model.fittedvalues),
                end=len(self.fitted_model.fittedvalues) + steps - 1,
                exog=exog
            ).conf_int()
            
            predictions = {
                'forecast': forecast_result,
                'confidence_intervals': pred_ci,
                'prediction_std': self.fitted_model.get_prediction(
                    start=len(self.fitted_model.fittedvalues),
                    end=len(self.fitted_model.fittedvalues) + steps - 1,
                    exog=exog
                ).se_mean
            }
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            raise
