from scripts.forecasting.pre_process_income_data import PreprocessNextWeekForecast


import pandas as pd
from prophet.serialize import model_from_json

import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_forecast(reference_date: str = None, model_path: str = '/opt/airflow/dags/models/prophet/prophet_with_metrics.json') -> pd.DataFrame:
    logger.info('Starting forecasting')
    
    pnwf = PreprocessNextWeekForecast(reference_date=reference_date)
    
    future_dataframe = pnwf.pipeline()
    logger.info(f'Future dataframe obtained')
    
    logger.info('Getting pre treined model.')
    
    with open(model_path, 'r') as fin:
        model = model_from_json(fin.read())    
    
    return model.predict(future_dataframe)
    
    
if __name__ == "__main__":
    print(run_forecast(model_path='models/prophet/prophet_with_metrics.json'))