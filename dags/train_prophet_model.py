import pendulum
import pandas as pd
import logging
import os

from airflow.decorators import dag, task
from airflow.models.param import Param


from scripts.forecaster_training.prophet_trainer import ProphetTrainer

logger = logging.getLogger(__name__)

DATA_FILE_PATH = '/opt/airflow/dags/data/energy_data.csv'

@dag(
    dag_id='train_energy_demand_prophet',
    start_date=pendulum.datetime(2025, 1, 1, tz="America/Sao_Paulo"),
    catchup=False,
    tags=['data-science', 'prophet', 'energy'],
    params={
        "model_name": Param(
            default="prophet_model_v1", 
            type="string", 
            description="Nome do arquivo do modelo (sem a extensão .json)"
        ),
        "output_dir": Param(
            default="/opt/airflow/dags/models/prophet/", 
            type="string", 
            description="Diretório onde o modelo será salvo"
        )
    }
)
def energy_forecast_pipeline():

    @task(task_id="train_model_task")
    def train_model(**kwargs):
        """
        Lê parâmetros, carrega dados e executa o pipeline de treino.
        """
        params = kwargs['params']
        model_name = params.get('model_name')
        output_dir = params.get('output_dir')
        
        logger.info(f"Iniciando pipeline. Modelo: {model_name} | Output: {output_dir}")

        if not os.path.exists(DATA_FILE_PATH):
            raise FileNotFoundError(f"Dataset não encontrado em: {DATA_FILE_PATH}")

        logger.info("Carregando dados do CSV...")
        df = pd.read_csv(DATA_FILE_PATH)

        logger.info("Inicializando ProphetTrainer...")
        trainer = ProphetTrainer(df=df, output_dir=output_dir)
        
        logger.info("Executando .train()...")
        trainer.train()

        logger.info("Salvando modelo...")
        trainer.save_model(model_name=model_name)
        
        return f"Modelo {model_name} salvo com sucesso em {output_dir}"

    train_model()

energy_forecast_pipeline()