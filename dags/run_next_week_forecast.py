from scripts.forecasting.run_forecast import run_forecast
from core.basic_configs import MLFLOW_TRACKING_URI

import logging
import os

import pendulum
from airflow.sdk import dag, task, Param
import mlflow


MLFLOW_EXPERIMENT = "Energy_Demand_Forecasting_Prophet"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dag(
    dag_id='next_week_forecast',
    start_date=pendulum.datetime(2025, 1, 1, tz='America/Sao_Paulo'),
    catchup=False,
    tags=['data-science', 'prophet', 'forecast', 'mlflow'],
    params={
        "reference_date": Param(
            default=None,
            type=["string", "null"],
            description='Reference date (YYYY-MM-DD). Uses today if None.'
        ),
        "pre_trained_model": Param(
            default='/opt/airflow/dags/models/prophet/prophet_with_metrics.json',
            type='string',
            description='Path to the pre-trained prophet model JSON'
        )
    }
)


def gen_forecast():
    """
    DAG to perform next week energy demand forecasting using a pre-trained Prophet model.
    Integrates with MLflow for experiment tracking and logging.
    """
    
    @task(task_id='forecasting')
    def perform_forecast(**kwargs):
        params = kwargs['params']
        ref_date = params.get('reference_date')
        model_path = params.get('pre_trained_model')
        
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        
        run_name = f"forecast_run_{kwargs['ds']}"
        
        logger.info(f"Starting MLflow Run: {run_name}")
        
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("reference_date", ref_date if ref_date else "today")
            mlflow.log_param("model_path", model_path)
            mlflow.log_param("execution_date", kwargs['ds'])
            
            logger.info("Executing forecast script...")
            forecast_df = run_forecast(reference_date=ref_date, model_path=model_path)
            
            output_file = f"forecast_{kwargs['ds']}.csv"
            forecast_df.to_csv(output_file, index=False)
            mlflow.log_artifact(output_file)
            
            production_path = "/opt/airflow/data/processed/latest_forecast.csv"
            os.makedirs(os.path.dirname(production_path), exist_ok=True)
            forecast_df.to_csv(production_path, index=False)
            logger.info(f"Production forecast file updated: {production_path}")

            if os.path.exists(output_file): os.remove(output_file)

    perform_forecast()

dag_instance = gen_forecast()