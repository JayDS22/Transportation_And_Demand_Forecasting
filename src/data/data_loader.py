import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging
from datetime import datetime, timedelta

class TransportationDataLoader:
    """
    Data loader for transportation demand forecasting
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def load_demand_data(self, file_path: str) -> pd.DataFrame:
        """Load historical demand data"""
        try:
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            df = df.sort_index()
            
            # Basic data validation
            assert 'demand' in df.columns, "Demand column missing"
            assert df.index.is_monotonic_increasing, "Timestamp not sorted"
            
            self.logger.info(f"Loaded {len(df)} demand records")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading demand data: {e}")
            raise
    
    def load_weather_data(self, file_path: str) -> pd.DataFrame:
        """Load weather data as external regressor"""
        try:
            weather_df = pd.read_csv(file_path)
            weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
            weather_df = weather_df.set_index('timestamp')
            
            # Feature engineering for weather
            weather_df['temp_celsius'] = (weather_df['temperature'] - 32) * 5/9
            weather_df['is_rainy'] = (weather_df['precipitation'] > 0).astype(int)
            weather_df['wind_speed_kmh'] = weather_df['wind_speed'] * 1.60934
            
            self.logger.info(f"Loaded {len(weather_df)} weather records")
            return weather_df
            
        except Exception as e:
            self.logger.error(f"Error loading weather data: {e}")
            raise
    
    def load_events_data(self, file_path: str) -> pd.DataFrame:
        """Load special events data"""
        try:
            events_df = pd.read_csv(file_path)
            events_df['date'] = pd.to_datetime(events_df['date'])
            
            # Create event impact features
            events_df['event_magnitude'] = events_df['expected_attendance'] / 10000
            events_df['is_major_event'] = (events_df['expected_attendance'] > 50000).astype(int)
            
            self.logger.info(f"Loaded {len(events_df)} event records")
            return events_df
            
        except Exception as e:
            self.logger.error(f"Error loading events data: {e}")
            raise
    
    def merge_datasets(self, demand_df: pd.DataFrame, 
                      weather_df: pd.DataFrame, 
                      events_df: pd.DataFrame) -> pd.DataFrame:
        """Merge all datasets with proper time alignment"""
        
        # Resample weather data to match demand frequency
        if self.config.get('resample_freq', 'H'):
            weather_resampled = weather_df.resample(self.config['resample_freq']).ffill()
        
        # Merge demand and weather
        merged_df = demand_df.join(weather_resampled, how='left')
        
        # Add event features
        merged_df['date'] = merged_df.index.date
        events_agg = events_df.groupby('date').agg({
            'event_magnitude': 'sum',
            'is_major_event': 'max'
        })
        
        merged_df = merged_df.join(events_agg, on='date', how='left')
        merged_df = merged_df.fillna(0)
        
        # Add holiday indicators
        merged_df = self._add_holiday_features(merged_df)
        
        self.logger.info(f"Merged dataset shape: {merged_df.shape}")
        return merged_df
    
    def _add_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add holiday and special day features"""
        df['is_weekend'] = df.index.weekday.isin([5, 6]).astype(int)
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_holiday'] = self._get_holiday_indicator(df.index)
        
        return df
    
    def _get_holiday_indicator(self, date_index) -> np.ndarray:
        """Get holiday indicators for dates"""
        # Simplified holiday detection - can be extended with holiday library
        holidays = [
            '2023-01-01', '2023-07-04', '2023-12-25',  # Major holidays
            '2024-01-01', '2024-07-04', '2024-12-25'
        ]
        holiday_dates = pd.to_datetime(holidays).date
        return np.isin(date_index.date, holiday_dates).astype(int)
