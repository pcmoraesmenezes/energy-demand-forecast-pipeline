import numpy as np
import pandas as pd
import mlflow
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error

import pendulum
import logging
import os


from airflow.sdk import dag, task, Param


from scripts.forecaster_training.prophet_trainer import ProphetTrainer
from core.basic_configs import MLFLOW_TRACKING_URI, DATA_DIR


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_FILE_PATH = os.path.join(DATA_DIR, 'energy_data.csv')


@dag(
    dag_id='train_energy_demand_prophet',
    start_date=pendulum.datetime(2025, 1, 1, tz="America/Sao_Paulo"),
    catchup=False,
    tags=['data-science', 'prophet', 'energy'],
    params={
        "model_name": Param(
            default="prophet_model_v1", 
            type="string", 
            description="Model name to register in MLflow (without .json extension)"
        ),
        "output_dir": Param(
            default="/opt/airflow/data/models/prophet/", 
            type="string", 
            description="Directory where the model will be saved"
        ),
        "run_description": Param(
            default="Training of Prophet model for energy demand forecasting.", 
            type="string", 
            description="MLflow run description"
        )
    }
)
def energy_forecast_pipeline():

    @task(task_id="train_model_task")
    def train_model(**kwargs):
        """
        Task to train a Prophet model for energy demand forecasting.
        Logs parameters and metrics to MLflow, and saves the trained model.
        """
        params = kwargs['params']
        model_name = params.get('model_name')
        output_dir = params.get('output_dir')
        run_description = params.get('run_description')
        
        logger.info(f"Initializing pipeline. Model: {model_name} | Output: {output_dir}")
        
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("Energy_Demand_Forecasting_Prophet")
        
        # mlflow.prophet.autolog(log_models=True)
        
        logging.info('Starting pipeline with mlflow tracking URI: %s', MLFLOW_TRACKING_URI)

        if not os.path.exists(DATA_FILE_PATH):
            raise FileNotFoundError(f"Dataset not found at: {DATA_FILE_PATH}")

        logger.info("Loading dataset from %s", DATA_FILE_PATH)
        df = pd.read_csv(DATA_FILE_PATH)
        
        with mlflow.start_run(run_name=f'run_{model_name}'):
            mlflow.log_param("data_source", DATA_FILE_PATH)
            mlflow.log_param("dataset_rows", df.shape[0])
            
            mlflow.set_tag("mlflow.note.content", run_description)
            mlflow.set_tag("model_type", "Prophet")
            mlflow.set_tag("trigger", "Airflow DAG")

            logger.info("initializing ProphetTrainer...")
            trainer = ProphetTrainer(df=df, output_dir=output_dir)
            
            logger.info("Training model...")
            trainer.train()
            
            logger.info("Collecting training metrics...")
            forecast = trainer.model.predict(trainer.processed_df)
            y_true = trainer.processed_df['y'].values
            y_pred = forecast['yhat'].values
            
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            non_zero_mask = y_true != 0
            if np.sum(non_zero_mask) > 0:
                mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
            else:
                mape = 0.0
                
                
            mlflow.log_metric("training_mae", mae)
            mlflow.log_metric("training_rmse", rmse)
            mlflow.log_metric("training_mape", mape)


            input_example = trainer.processed_df.head(5)
            signature = infer_signature(input_example, forecast.head(5))
            
            mlflow.prophet.log_model(
                trainer.model, 
                artifact_path="model",
                registered_model_name=model_name,
                signature=signature,
                input_example=input_example
            )
            
            mlflow.log_params({
                "growth": trainer.model.growth,
                "seasonality_mode": trainer.model.seasonality_mode,
                "interval_width": trainer.model.interval_width,
                "changepoint_prior_scale": trainer.model.changepoint_prior_scale
            })

            logger.info("Saving model")
            trainer.save_model(model_name=model_name)
            
        return f"Model saved in {output_dir}"

    train_model()

energy_forecast_pipeline()