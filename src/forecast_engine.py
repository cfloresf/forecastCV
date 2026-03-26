import pandas as pd
import numpy as np
import lightgbm as lgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Tuple, Any

class ForecastEngine:
    def __init__(self, df: pd.DataFrame, target_col: str = 'value'):
        """
        Initializes the engine with historical data.
        df must have a datetime index.
        """
        self.df = df.copy()
        if not isinstance(self.df.index, pd.DatetimeIndex):
            if 'date' in self.df.columns:
                self.df['date'] = pd.to_datetime(self.df['date'])
                self.df = self.df.set_index('date')
        
        self.target_col = target_col
        self.models = {}

    def prepare_features_lgb(self, df_input: pd.DataFrame, lags: List[int] = [1, 2, 3, 7, 14]) -> pd.DataFrame:
        """
        Engineers features for LightGBM (lags, rolling stats, date parts).
        """
        df = df_input.copy()
        for lag in lags:
            df[f'lag_{lag}'] = df[self.target_col].shift(lag)
        
        df['rolling_mean_7'] = df[self.target_col].shift(1).rolling(window=7).mean()
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['dayofweek'] = df.index.dayofweek
        
        return df.dropna()

    def train_sarima(self, order=(1,1,1), seasonal_order=(0,1,1,12)):
        """
        Trains a SARIMA model.
        """
        try:
            model = SARIMAX(self.df[self.target_col], order=order, seasonal_order=seasonal_order, 
                             enforce_stationarity=False, enforce_invertibility=False)
            self.models['sarima'] = model.fit(disp=False)
            return True, "SARIMA Trained"
        except Exception as e:
            return False, str(e)

    def train_lightgbm(self):
        """
        Trains a LightGBM regressor on lagged features.
        """
        df_feat = self.prepare_features_lgb(self.df)
        X = df_feat.drop(columns=[self.target_col])
        y = df_feat[self.target_col]
        
        # Simple split or full train for demo
        model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, 
                                   importance_type='gain', verbose=-1)
        model.fit(X, y)
        self.models['lgb'] = model
        return True, "LightGBM Trained"

    def forecast(self, horizon: int = 12, method: str = 'sarima') -> pd.DataFrame:
        """
        Predicts future values for the given horizon.
        """
        last_date = self.df.index[-1]
        freq = self.df.index.inferred_freq or 'M'
        future_dates = pd.date_range(start=last_date + pd.tseries.frequencies.to_offset(freq), 
                                      periods=horizon, freq=freq)
        
        if method == 'sarima' and 'sarima' in self.models:
            forecast_obj = self.models['sarima'].get_forecast(steps=horizon)
            pred = forecast_obj.predicted_mean
            conf_int = forecast_obj.conf_int()
            
            res = pd.DataFrame({
                'date': future_dates,
                'forecast': pred.values,
                'lower_ci': conf_int.iloc[:, 0].values,
                'upper_ci': conf_int.iloc[:, 1].values
            })
            return res
            
        elif method == 'lgb' and 'lgb' in self.models:
            # Recursive forecasting for LGB
            current_data = self.df.copy()
            forecasts = []
            
            for i in range(horizon):
                feat_df = self.prepare_features_lgb(current_data)
                X_last = feat_df.tail(1).drop(columns=[self.target_col])
                pred = self.models['lgb'].predict(X_last)[0]
                forecasts.append(pred)
                
                # Update current data with new prediction to get lags for next step
                new_row = pd.DataFrame({self.target_col: [pred]}, index=[future_dates[i]])
                current_data = pd.concat([current_data, new_row])
                
            return pd.DataFrame({'date': future_dates, 'forecast': forecasts})
            
        return pd.DataFrame()
