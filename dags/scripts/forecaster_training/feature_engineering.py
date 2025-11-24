from core.basic_configs import WEATHER_LOCATION, BASE_TEMP_API, PANDEMIC_START_DATE, PANDEMIC_END_DATE


import pandas as pd
import logging
import holidays
import openmeteo_requests
import requests_cache
from retry_requests import retry


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
        
    def _prophet_data_structure(self) -> pd.DataFrame:
       logger.info("Converting year and week to datetime format for Prophet model.")
       
       ds = pd.DataFrame()
       ds['ds'] = pd.to_datetime(self.df['year'].astype(str) + self.df['week'].astype(str) + '1', format='%Y%W%w')
       ds['y'] = self.df['val_cargaenergiahomwmed']
       return ds
   
   
    def _add_holidays_brazil(self, df: pd.DataFrame) -> pd.DataFrame:
       logger.info('Adding brazillian holidays to the dataset.')
       
       br_holidays = holidays.BR(categories={"public", "optional"})
       sp_holidays = holidays.BR(subdiv="SP")
       
       logger.info('Combining holidays from Brazil and São Paulo state.')
       
       combined_holidays = br_holidays + sp_holidays
       
       ds_df = df.copy()
        
       ds_df['is_holiday_week'] = ds_df['ds'].apply(
            lambda x: 1 if any((x + pd.Timedelta(days=i)) in combined_holidays for i in range(7)) else 0
        )
        
       return ds_df       
   
   
    def _pandemic_period_flag(self, df: pd.DataFrame) -> pd.DataFrame:
         logger.info('Adding pandemic period flag to the dataset.')
         
         df['is_pandemic_period'] = df['ds'].apply(
             lambda x: 1 if PANDEMIC_START_DATE <= x.strftime('%Y-%m-%d') <= PANDEMIC_END_DATE else 0
         )
         
         return df
   
   
    def _get_historical_weather(self):
        lats = [loc['lat'] for loc in WEATHER_LOCATION]
        longs = [loc['long'] for loc in WEATHER_LOCATION]
        
        cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)
        
        min_date = pd.to_datetime(f"{self.df['year'].min()}-01-01")
        
        dataset_end_of_year = pd.to_datetime(f"{self.df['year'].max()}-12-31")
        
        today = pd.to_datetime("today").normalize()
        
        final_end_date = min(dataset_end_of_year, today)
        
        str_min_date = min_date.strftime('%Y-%m-%d')
        str_max_date = final_end_date.strftime('%Y-%m-%d')
        
        logger.info(f"Requesting weather data from {str_min_date} to {str_max_date}")

        params = {
            "latitude": lats,
            "longitude": longs,
            "start_date": str_min_date,
            "end_date": str_max_date, 
            "daily": "temperature_2m_max",
            "timezone": "America/Sao_Paulo"
        }
        
        responses = openmeteo.weather_api(BASE_TEMP_API, params=params)
        combined_temps = pd.DataFrame()
        
        for i, response in enumerate(responses):
            loc_weight = WEATHER_LOCATION[i]['weight']
            loc_name = WEATHER_LOCATION[i]['name']
            
            daily = response.Daily()
            temps = daily.Variables(0).ValuesAsNumpy()
            
            dates = pd.date_range(
                start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
                end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
                freq = pd.Timedelta(seconds = daily.Interval()),
                inclusive = "left"
            )
            
            if combined_temps.empty:
                combined_temps['date'] = dates
                combined_temps.set_index('date', inplace=True)
            
            if len(temps) > len(combined_temps):
                temps = temps[:len(combined_temps)]
            elif len(temps) < len(combined_temps):
                combined_temps = combined_temps.iloc[:len(temps)]

            combined_temps[f'temp_weighted_{loc_name}'] = temps * loc_weight

        combined_temps['temp_brasil_index'] = combined_temps.sum(axis=1)
        
        combined_temps.reset_index(inplace=True)
        
        combined_temps['date'] = combined_temps['date'].dt.tz_convert(None) 
        
        weekly_weather = combined_temps.resample('W-MON', on='date')[['temp_brasil_index']].mean().reset_index()
        
        weekly_weather.rename(columns={'date': 'ds', 'temp_brasil_index': 'temp_max_br'}, inplace=True)
        
        return weekly_weather
    

    def pipeline(self) -> pd.DataFrame:
        logger.info("Starting feature engineering pipeline.")
        
        prophet_df = self._prophet_data_structure()
        prophet_df = self._add_holidays_brazil(prophet_df)
        
        weather_df = self._get_historical_weather()
        
        pandemic = self._pandemic_period_flag(prophet_df)
        prophet_df['is_pandemic_period'] = pandemic['is_pandemic_period']
        
        logger.info("Merging weather data with main dataframe.")
        
        final_df = pd.merge(prophet_df, weather_df, on='ds', how='left')
        
        logger.info("Feature engineering pipeline completed.")
        
        return final_df
    
    
    
if __name__ == "__main__":
    df = pd.read_csv('data/energy_data.csv')
    fe = FeatureEngineer(df)
    final_df = fe.pipeline()
    print(final_df.head())
    print(final_df.info())
    print(final_df[final_df['is_holiday_week'] != 0])
    print(final_df[final_df['is_pandemic_period'] != 0])