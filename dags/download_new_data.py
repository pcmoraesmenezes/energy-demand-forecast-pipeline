import os
import pandas as pd
import pendulum
import logging

from airflow.decorators import dag, task

from core.basic_configs import DATASET_PATH
from scripts.data_handler.pre_process_data import update_new_data 

logger = logging.getLogger("airflow.task")

default_args = {
    'owner': 'airflow',
    'retries': 1,
}

@dag(
    dag_id='update_energy_data_weekly',
    default_args=default_args,
    schedule='0 8 * * 1', 
    start_date=pendulum.today('UTC').add(days=-1),    
    catchup=False, 
    tags=['energy', 'etl']
)
def energy_update_pipeline():

    @task
    def run_update_process():
        """
        Main function to update the energy dataset with new weekly data.
        """
        now = pendulum.now('America/Sao_Paulo')
        current_year = now.year
        
        iso_week = now.isocalendar().week
        
        logger.info(f"Starting pipeline. Current context: Year {current_year}, Week {iso_week}")
        logger.info(f"Target Dataset Path: {DATASET_PATH}")

        if os.path.exists(DATASET_PATH):
            logger.info("Existing dataset found. Loading...")
            try:
                existing_df = pd.read_csv(DATASET_PATH)
            except Exception as e:
                logger.error(f"Error reading CSV: {e}")
                raise e
        else:
            logger.info("No existing dataset found. Creating a new empty DataFrame structure.")
            existing_df = pd.DataFrame(columns=['year', 'week', 'val_cargaenergiahomwmed'])

        update_new_data(existing_data=existing_df, year=current_year, file_path=DATASET_PATH)
        
        logger.info("Update process completed successfully.")

    run_update_process()

energy_update_pipeline()