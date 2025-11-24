from scripts.data_handler.energy_downloader import download_energy_data


import pandas as pd
import logging
import os


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def update_new_data(existing_data: pd.DataFrame, year: int, file_path: str = '/opt/airflow/dags/data/energy_data.csv') -> None:
    logger.info(f'Updating data for year: {year}')
    
    last_week = 0
    
    if not existing_data.empty:
        subset_year = existing_data[existing_data['year'] == year]
        
        if not subset_year.empty:
            last_week = subset_year['week'].max()
            logger.info(f'Last week in existing data for year {year}: {last_week}')
        else:
            logger.info(f'Year {year} not found in existing data. Starting from week 0.')
    else:
        logger.info('Existing data is empty. Starting from scratch.')

    new_data_full = pre_build_new_data(year)
    
    new_records = new_data_full[new_data_full['week'] > last_week]
    
    if new_records.empty:
        logger.info(f'No new data found for year {year} after week {last_week}.')
        return None

    logger.info(f'Found {len(new_records)} new records to append.')

    updated_df = pd.concat([existing_data, new_records], ignore_index=True)
    
    updated_df = updated_df.sort_values(by=['year', 'week'])

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    try:
        updated_df.to_csv(file_path, index=False)
        logger.info(f'Successfully saved updated dataset to: {file_path}')
    except Exception as e:
        logger.error(f'Failed to save file at {file_path}. Error: {e}')
        raise e

    return None
        
        
def pre_build_new_data(year: int) -> pd.DataFrame:
    
    logger.info(f'Pre-building new data for year: {year}')
    data = download_energy_data(year)
    
    data['din_instante'] = pd.to_datetime(data['din_instante'])
    
    iso_data = data['din_instante'].dt.isocalendar()
    data['year'] = iso_data.year
    data['week'] = iso_data.week
    
    weekly_data = data.groupby(['year', 'week'])['val_cargaenergiahomwmed'].sum().reset_index()
    
    logger.info('Data successfully pre-processed to weekly format with year metadata')
    
    df = weekly_data[['year', 'week', 'val_cargaenergiahomwmed']]
    return df
    