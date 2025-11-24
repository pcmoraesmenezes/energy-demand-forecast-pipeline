from scripts.forecaster_training.feature_engineering import FeatureEngineer


import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json

import logging
import os


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProphetTrainer:
    def __init__(self, df: pd.DataFrame = '/opt/airflow/dags/data/energy_data.csv', output_dir: str = '/opt/airflow/dags/models/prophet/'):
        logger.info("Initializing ProphetTrainer.")
        
        self.output_dir = output_dir
        
        self.df = df.copy()
        logger.info(f"DataFrame copied with shape: {self.df.shape}")
        
        self.model = Prophet()
        logger.info("Prophet model initialized.")
        
        self.fe = FeatureEngineer(self.df)
        self.processed_df = self.fe.pipeline()
        logger.info(f"Feature engineering completed. Processed DataFrame shape: {self.processed_df.shape}")
        
        self._add_regressor()
        logger.info("Regressors added to the Prophet model.")
        
    def _add_regressor(self):
        logger.info('adding regressors to the Prophet model.')
        
        regressor_cols = [col for col in self.processed_df.columns if col not in ['ds', 'y']]
        for col in regressor_cols:
            self.model.add_regressor(col)
            logger.info(f"Regressor added: {col}")
            
    
    def train(self):
        logger.info("Training the Prophet model.")
        self.model.fit(self.processed_df)
        logger.info("Model training completed.")
            
    
    def save_model(self, model_name: str = 'prophet_model'):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Output directory created: {self.output_dir}")
            
        model_path = os.path.join(self.output_dir, f"{model_name}.json")
        logger.info(f"Saving model to {model_path}")
        
        with open(model_path, 'w') as f:
            f.write(model_to_json(self.model))        
            
        logger.info("Model saved successfully.")
        
        

if __name__ == "__main__":
    df = pd.read_csv('data/energy_data.csv')
    output_dir = 'models/prophet/'
    trainer = ProphetTrainer(df, output_dir)
    trainer.train()
    trainer.save_model()
        