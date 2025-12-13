# Energy Demand Forecast Pipeline

This project implements an end-to-end Machine Learning pipeline designed to forecast the weekly energy load demand for the Brazilian territory. The architecture prioritizes MLOps best practices, including automated data ingestion, model versioning, experiment tracking, and a visualization dashboard.

The core objective is to demonstrate a robust architecture for Model Lifecycle Management rather than maximizing statistical accuracy through extensive hyperparameter tuning.

## Architecture Overview

The system operates as a fully containerized environment orchestrated by Docker.

1.  **Orchestration (Airflow):** Manages the ETL process, model training triggers, and forecasting schedules.
2.  **Experiment Tracking (MLflow):** Tracks model parameters, metrics (MAE, RMSE), and artifacts. Acts as the Model Registry.
3.  **Forecasting Engine:** Uses Facebook Prophet with external regressors (weather data, holidays, and pandemic flags).
4.  **Visualization (Streamlit):** Consumes the latest forecast and historical data to present a business-facing dashboard.

## Tech Stack

* **Python 3.12**
* **Apache Airflow 3.1.3** (Orchestration)
* **MLflow** (Experiment Tracking & Model Registry)
* **Streamlit** (Data Visualization)
* **Facebook Prophet** (Time Series Forecasting)
* **Docker & Docker Compose** (Containerization)
* **PostgreSQL & Redis** (Backend services)

## Project Structure

```text
.
├── dags/                       # Airflow DAGs (ETL, Training, Forecasting)
│   ├── scripts/                # Core logic for data handling and modeling
│   └── models/                 # Serialized model artifacts
├── streamlit_app/              # Dashboard application source code
├── docker-compose.yaml         # Container orchestration config
└── requirements.txt            # Python dependencies
````

## Getting Started

### Prerequisites

  * Docker Engine
  * Docker Compose

### Installation & Execution

1.  Clone the repository:

    ```bash
    git clone https://github.com/pcmoraesmenezes/energy-demand-forecast-pipeline
    cd energy-demand-forecast-pipeline
    ```

2.  Build and start the services:

    ```bash
    docker-compose up -d --build
    ```

    *Note: This command builds the custom Airflow and Streamlit images and starts Postgres, Redis, MLflow, Airflow, and the Dashboard.*

3.  Verify running containers:

    ```bash
    docker-compose ps
    ```

## Accessing Services

Once the containers are up and healthy, access the interfaces via your browser:

| Service | URL | Default Credentials |
| :--- | :--- | :--- |
| **Apache Airflow** | `http://localhost:8080` | User: `airflow` / Pass: `airflow` |
| **MLflow UI** | `http://localhost:5000` | N/A |
| **Streamlit Dashboard** | `http://localhost:8501` | N/A |

## Pipelines (DAGs)

The system is driven by three main Airflow DAGs:

1.  **`update_energy_data_weekly`**:

      * Runs weekly (Mondays).
      * Downloads the latest energy load CSV from the ONS (Operador Nacional do Sistema Elétrico) open data portal.
      * Pre-processes and appends new data to the local dataset.

2.  **`train_energy_demand_prophet`**:

      * Triggered to retrain the model with the most recent data.
      * Performs feature engineering (weather integration via OpenMeteo API, holiday flagging).
      * Trains a Prophet model and logs metrics/artifacts to MLflow.

3.  **`next_week_forecast`**:

      * Loads the latest production model from MLflow.
      * Generates the forecast for the upcoming week.
      * Updates the data source used by the Streamlit dashboard.

## Dashboard Features

The Streamlit application serves as the consumption layer for the pipeline results. Key features include:

  * **KPI Monitoring:** Displays current model version, training MAE/RMSE, and forecasted load.
  * **Interactive Plot:** Visualizes historical data vs. predicted values with confidence intervals.
  * **Data Export:** Allows users to download the forecast data as CSV.
  * **MLflow Integration:** dynamically fetches model metadata (version and metrics) directly from the tracking server.

## Model & Limitations

**Model Used:** Facebook Prophet.
**Features:**

  * `ds` (Date), `y` (Energy Load).
  * **Regressors:** Weighted average temperature (Brazil), National Holidays, Pandemic Period Flag.

**Disclaimer:**
This project focuses on the **MLOps pipeline construction**, containerization, and tool integration (Airflow/MLflow). Extensive hyperparameter tuning, model selection comparison, and deep statistical optimization were explicitly out of scope. The model parameters are set to reasonable defaults to demonstrate the flow.
