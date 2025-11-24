from core.basic_configs import BASE_URL, EXTENSION

import pandas as pd
import requests
import io


import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_url(desired_year: int) -> str:
    logger.info(f'Building URL for year: {desired_year}')
    return f'{BASE_URL}{desired_year}{EXTENSION}'


def download_energy_data(desired_year: int) -> pd.DataFrame:
    url = build_url(desired_year)
    logger.info(f'Downloading data from URL: {url}')
    
    response = requests.get(url)
    logger.info(f'Download completed with status code: {response.status_code}')
    response.raise_for_status()
    
    logger.info('Reading data into DataFrame')
    
    csv_content = response.content.decode('utf-8')
    data = pd.read_csv(io.StringIO(csv_content), sep=';')
    
    
    logger.info('Data successfully read into DataFrame')
    
    return data