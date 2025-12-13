from core.basic_configs import WEATHER_LOCATION, BASE_FORECAST_API, PANDEMIC_START_DATE, PANDEMIC_END_DATE


import pandas as pd
from datetime import timedelta
import logging

import holidays

import openmeteo_requests
import requests_cache
from retry_requests import retry




logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreprocessNextWeekForecast:
    def __init__(self, reference_date: str = None):
        
        if reference_date:
            self.ref_date = pd.to_datetime(reference_date)
        else:
            self.ref_date = pd.to_datetime("today").normalize()
            
    def _generate_future_ds(self) -> pd.DataFrame:
        logger.info("Generating 'ds' for the next week.")
        
        days_ahead = 7 - self.ref_date.weekday()
        if days_ahead <= 0: 
             days_ahead += 7
             
        next_monday = self.ref_date + timedelta(days=days_ahead)
        
        df = pd.DataFrame({'ds': [next_monday]})
        return df

    def _add_holidays_brazil(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info('Calculating holiday flag for the forecast week.')
        
        br_holidays = holidays.BR(categories={"public", "optional"})
        sp_holidays = holidays.BR(subdiv="SP")
        combined_holidays = br_holidays + sp_holidays
        
        ds_val = df['ds'].iloc[0]
        
        is_holiday = 1 if any((ds_val + timedelta(days=i)) in combined_holidays for i in range(7)) else 0
        
        df['is_holiday_week'] = is_holiday
        return df

    def _pandemic_period_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info('Calculating pandemic period flag.')
        
        ds_val = df['ds'].iloc[0]
        
        is_pandemic = 1 if PANDEMIC_START_DATE <= ds_val.strftime('%Y-%m-%d') <= PANDEMIC_END_DATE else 0
        
        df['is_pandemic_period'] = is_pandemic
        return df

    def _get_forecast_weather(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Fetching weather forecast data for the next week.")
        
        ds_val = df['ds'].iloc[0]
        start_date_str = ds_val.strftime('%Y-%m-%d')
        end_date_str = (ds_val + timedelta(days=6)).strftime('%Y-%m-%d')
        
        cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)
        
        lats = [loc['lat'] for loc in WEATHER_LOCATION]
        longs = [loc['long'] for loc in WEATHER_LOCATION]
        

        params = {
            "latitude": lats,
            "longitude": longs,
            "start_date": start_date_str,
            "end_date": end_date_str,
            "daily": "temperature_2m_max",
            "timezone": "America/Sao_Paulo"
        }
        
        responses = openmeteo.weather_api(BASE_FORECAST_API, params=params)
        
        temp_daily_df = pd.DataFrame()
        
        dates = pd.date_range(start=ds_val, periods=7, freq='D')
        temp_daily_df['date'] = dates
        temp_daily_df.set_index('date', inplace=True)

        for i, response in enumerate(responses):
            loc_weight = WEATHER_LOCATION[i]['weight']
            loc_name = WEATHER_LOCATION[i]['name']
            
            daily = response.Daily()
            temps = daily.Variables(0).ValuesAsNumpy()
            
            if len(temps) > 7: temps = temps[:7]
            
            temp_daily_df[f'temp_weighted_{loc_name}'] = temps * loc_weight
            
        temp_daily_df['temp_brasil_index'] = temp_daily_df.sum(axis=1)
        
        avg_weekly_temp = temp_daily_df['temp_brasil_index'].mean()
        
        df['temp_max_br'] = avg_weekly_temp
        
        return df
    

    def pipeline(self) -> pd.DataFrame:
        logger.info("Starting inference preparation pipeline.")
        
        future_df = self._generate_future_ds()
        
        future_df = self._add_holidays_brazil(future_df)
        future_df = self._pandemic_period_flag(future_df)
        
        future_df = self._get_forecast_weather(future_df)
        
        logger.info("Inference dataframe ready.")
        return future_df