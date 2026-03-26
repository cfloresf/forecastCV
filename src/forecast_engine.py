import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any

class ForecastEngine:
    def __init__(self, df: pd.DataFrame, target_col: str = 'value'):
        self.df = df.copy()
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df = self.df.set_index('date').sort_index()
        self.target_col = target_col

    def forecast(self, horizon: int = 6, method: str = 'linear') -> pd.DataFrame:
        """
        Performs forecasting using pure NumPy matrix math (Least Squares).
        No scipy/statsmodels/lightgbm required.
        """
        y = self.df[self.target_col].values
        x = np.arange(len(y))
        
        # Degree 1 for linear, Degree 2 for 'growth' trend
        degree = 1 if method == 'linear' else 2
        coeffs = np.polyfit(x, y, degree)
        model = np.poly1d(coeffs)
        
        # Predict future
        future_x = np.arange(len(y), len(y) + horizon)
        forecast_values = model(future_x)
        
        # Dates
        last_date = self.df.index[-1]
        freq = self.df.index.inferred_freq or 'M'
        future_dates = pd.date_range(start=last_date + pd.tseries.frequencies.to_offset(freq), 
                                      periods=horizon, freq=freq)
        
        # Calculate simple confidence intervals (Standard Deviation of residuals)
        residuals = y - model(x)
        std_err = np.std(residuals)
        
        return pd.DataFrame({
            'date': future_dates,
            'forecast': forecast_values,
            'lower_ci': forecast_values - (1.96 * std_err),
            'upper_ci': forecast_values + (1.96 * std_err)
        })
