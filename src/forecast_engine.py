import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any

class ForecastEngine:
    def __init__(self, df: pd.DataFrame, target_col: str = 'value'):
        """Initialize with data validation."""
        self.df = df.copy()
        if 'date' not in self.df.columns:
            raise ValueError("DataFrame must contain 'date' column")
        if target_col not in self.df.columns:
            raise ValueError(f"DataFrame must contain '{target_col}' column")
        
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date')
        
        # Validate data quality
        if self.df[target_col].isna().sum() > 0:
            print(f"⚠️  Found {self.df[target_col].isna().sum()} missing values, using forward fill")
            self.df[target_col] = self.df[target_col].fillna(method='ffill').fillna(method='bfill')
        
        if len(self.df) < 3:
            raise ValueError(f"Need at least 3 data points for forecasting, got {len(self.df)}")
        
        self.df = self.df.set_index('date').sort_index()
        self.target_col = target_col

    def forecast(self, horizon: int = 6, method: str = 'linear') -> pd.DataFrame:
        """
        Performs forecasting using pure NumPy matrix math (Least Squares).
        Includes validation and error handling.
        """
        if horizon < 1:
            raise ValueError("Horizon must be at least 1")
        if horizon > 120:
            print(f"⚠️  Horizon {horizon} is very large. Consider reducing for better accuracy.")
        
        try:
            y = self.df[self.target_col].values.astype(float)
            x = np.arange(len(y))
            
            # Validate data
            if np.isnan(y).any() or np.isinf(y).any():
                raise ValueError("Data contains NaN or Inf values")
            
            # Degree 1 for linear, Degree 2 for 'growth' trend
            degree = 1 if method == 'linear' else 2
            coeffs = np.polyfit(x, y, degree)
            model = np.poly1d(coeffs)
            
            # Predict future
            future_x = np.arange(len(y), len(y) + horizon)
            forecast_values = model(future_x)
            
            # Dates
            last_date = self.df.index[-1]
            freq = self.df.index.inferred_freq or 'D'
            
            try:
                future_dates = pd.date_range(
                    start=last_date + pd.tseries.frequencies.to_offset(freq), 
                    periods=horizon, 
                    freq=freq
                )
            except Exception as e:
                print(f"⚠️  Could not infer frequency ({e}), using 'D' (daily)")
                future_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=horizon,
                    freq='D'
                )
            
            # Calculate confidence intervals (Standard Deviation of residuals)
            residuals = y - model(x)
            std_err = np.std(residuals)
            
            result = pd.DataFrame({
                'date': future_dates,
                'forecast': forecast_values,
                'lower_ci': forecast_values - (1.96 * std_err),
                'upper_ci': forecast_values + (1.96 * std_err)
            })
            
            # Ensure no NaN in output
            if result.isna().any().any():
                print("⚠️  Found NaN in forecast results, replacing with forward fill")
                result = result.fillna(method='ffill')
            
            return result
        except Exception as e:
            raise ValueError(f"Forecasting failed: {e}")
