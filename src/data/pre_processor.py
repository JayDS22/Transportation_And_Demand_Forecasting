import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from typing import Tuple, Optional, Dict, Any, List
import logging

class TimeSeriesPreprocessor:
    """
    Preprocessing pipeline for time series forecasting
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scalers = {}
        self.imputers = {}
        self.logger = logging.getLogger(__name__)
        
    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Fit preprocessors and transform data"""
        
        # Handle missing values
        df_imputed = self._handle_missing_values(df)
        
        # Feature engineering
        df_features = self._engineer_features(df_imputed)
        
        # Outlier detection and treatment
        df_clean = self._handle_outliers(df_features)
        
        # Scaling
        df_scaled = self._scale_features(df_clean)
        
        # Create sequences for LSTM
        sequences, metadata = self._create_sequences(df_scaled)
        
        preprocessing_info = {
            'scalers': self.scalers,
            'imputers': self.imputers,
            'feature_names': list(df_scaled.columns),
            'sequence_length': self.config.get('sequence_length', 24),
            'target_column': self.config.get('target_column', 'demand')
        }
        
        return sequences, preprocessing_info
    
    def transform(self, df: pd.DataFrame, preprocessing_info: Dict) -> np.ndarray:
        """Transform new data using fitted preprocessors"""
        
        # Apply same preprocessing steps
        df_imputed = self._apply_imputation(df)
        df_features = self._engineer_features(df_imputed)
        df_scaled = self._apply_scaling(df_features)
        sequences, _ = self._create_sequences(df_scaled)
        
        return sequences
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with appropriate strategies"""
        
        df_copy = df.copy()
        
        # Different strategies for different column types
        numeric_columns = df_copy.select_dtypes(include=[np.number]).columns
        categorical_columns = df_copy.select_dtypes(include=['object']).columns
        
        # Numeric columns - use forward fill then median
        for col in numeric_columns:
            if df_copy[col].isnull().sum() > 0:
                # Forward fill for time series continuity
                df_copy[col] = df_copy[col].fillna(method='ffill')
                
                # Median imputation for remaining nulls
                imputer = SimpleImputer(strategy='median')
                df_copy[[col]] = imputer.fit_transform(df_copy[[col]])
                self.imputers[col] = imputer
        
        # Categorical columns - mode imputation
        for col in categorical_columns:
            if df_copy[col].isnull().sum() > 0:
                imputer = SimpleImputer(strategy='most_frequent')
                df_copy[[col]] = imputer.fit_transform(df_copy[[col]])
                self.imputers[col] = imputer
        
        self.logger.info("Missing value imputation completed")
        return df_copy
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer time-based and lag features"""
        
        df_copy = df.copy()
        
        # Lag features
        target_col = self.config.get('target_column', 'demand')
        lag_periods = self.config.get('lag_periods', [1, 2, 3, 6, 12, 24])
        
        for lag in lag_periods:
            df_copy[f'{target_col}_lag_{lag}'] = df_copy[target_col].shift(lag)
        
        # Rolling statistics
        rolling_windows = self.config.get('rolling_windows', [3, 6, 12, 24])
        
        for window in rolling_windows:
            df_copy[f'{target_col}_rolling_mean_{window}'] = (
                df_copy[target_col].rolling(window=window).mean()
            )
            df_copy[f'{target_col}_rolling_std_{window}'] = (
                df_copy[target_col].rolling(window=window).std()
            )
        
        # Cyclical features
        if 'hour' in df_copy.columns:
            df_copy['hour_sin'] = np.sin(2 * np.pi * df_copy['hour'] / 24)
            df_copy['hour_cos'] = np.cos(2 * np.pi * df_copy['hour'] / 24)
        
        if 'day_of_week' in df_copy.columns:
            df_copy['dow_sin'] = np.sin(2 * np.pi * df_copy['day_of_week'] / 7)
            df_copy['dow_cos'] = np.cos(2 * np.pi * df_copy['day_of_week'] / 7)
        
        if 'month' in df_copy.columns:
            df_copy['month_sin'] = np.sin(2 * np.pi * df_copy['month'] / 12)
            df_copy['month_cos'] = np.cos(2 * np.pi * df_copy['month'] / 12)
        
        # Interaction features
        if 'temp_celsius' in df_copy.columns and 'is_weekend' in df_copy.columns:
            df_copy['temp_weekend_interaction'] = (
                df_copy['temp_celsius'] * df_copy['is_weekend']
            )
        
        self.logger.info(f"Feature engineering completed. New shape: {df_copy.shape}")
        return df_copy
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers using IQR method"""
        
        df_copy = df.copy()
        numeric_columns = df_copy.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col != self.config.get('target_column', 'demand'):  # Don't clip target
                Q1 = df_copy[col].quantile(0.25)
                Q3 = df_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Clip outliers
                df_copy[col] = np.clip(df_copy[col], lower_bound, upper_bound)
        
        self.logger.info("Outlier treatment completed")
        return df_copy
    
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale features using StandardScaler"""
        
        df_copy = df.copy()
        numeric_columns = df_copy.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            scaler = StandardScaler()
            df_copy[[col]] = scaler.fit_transform(df_copy[[col]])
            self.scalers[col] = scaler
        
        self.logger.info("Feature scaling completed")
        return df_copy
    
    def _create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """Create sequences for LSTM input"""
        
        sequence_length = self.config.get('sequence_length', 24)
        target_column = self.config.get('target_column', 'demand')
        
        # Remove rows with NaN values
        df_clean = df.dropna()
        
        # Prepare feature matrix
        feature_columns = [col for col in df_clean.columns if col != target_column]
        X = df_clean[feature_columns].values
        y = df_clean[target_column].values
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X)):
            X_sequences.append(X[i-sequence_length:i])
            y_sequences.append(y[i])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        metadata = {
            'n_samples': len(X_sequences),
            'sequence_length': sequence_length,
            'n_features': X_sequences.shape[2],
            'feature_names': feature_columns
        }
        
        self.logger.info(f"Created {len(X_sequences)} sequences")
        return (X_sequences, y_sequences), metadata
