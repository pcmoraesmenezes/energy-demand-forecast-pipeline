BASE_URL = ' https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/curva-carga-ho/CURVA_CARGA_'
EXTENSION = '.csv'
DATASET_PATH = '/opt/airflow/dags/data/energy_data.csv'

WEATHER_LOCATION = [
            {'name': 'SP', 'lat': -23.55, 'long': -46.63, 'weight': 0.35},
            {'name': 'RJ', 'lat': -22.90, 'long': -43.17, 'weight': 0.20},
            {'name': 'MG', 'lat': -19.91, 'long': -43.93, 'weight': 0.10}, 
            {'name': 'RS', 'lat': -30.03, 'long': -51.22, 'weight': 0.10}, 
            {'name': 'BA', 'lat': -12.97, 'long': -38.50, 'weight': 0.10}, 
            {'name': 'DF', 'lat': -15.79, 'long': -47.88, 'weight': 0.05}, 
            {'name': 'AM', 'lat': -3.11,  'long': -60.02, 'weight': 0.05}, 
            {'name': 'PE', 'lat': -8.04,  'long': -34.87, 'weight': 0.05}, 
]

BASE_TEMP_API = 'https://archive-api.open-meteo.com/v1/archive'

PANDEMIC_START_DATE = '2020-03-11'
PANDEMIC_END_DATE = '2023-05-05'