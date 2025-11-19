import pendulum
from airflow import DAG

from airflow.providers.standard.operators.python import PythonOperator


def print_hello():
    return 'Hello World'

with DAG(
    dag_id='hello_world',
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    schedule='@daily',
    catchup=False,
    tags=['teste']
) as dag:
    
    task = PythonOperator(
        task_id='print_hello_task',
        python_callable=print_hello,
    )