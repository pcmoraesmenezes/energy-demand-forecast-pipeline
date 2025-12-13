from datetime import timedelta
import os

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

import mlflow
from mlflow.tracking import MlflowClient


HISTORICAL_DATA_PATH = "/app/data/energy_data.csv"
FORECAST_DATA_PATH = "/app/data/processed/latest_forecast.csv"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = "Energy_Demand_Forecasting_Prophet"


st.set_page_config(
    page_title="Energy Demand Forecast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background-color: #0E1117;
    }
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }
    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    .metric-card-container {
        background-color: #161B22;
        border: 1px solid #30363D;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-label {
        font-size: 14px;
        color: #8B949E;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #E6EDF3;
    }
    .metric-delta {
        font-size: 14px;
        color: #3FB950;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #E6EDF3;
    }
    .stDataFrame {
        border: 1px solid #30363D;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)



@st.cache_data(ttl=3600)
def load_combined_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and combine historical and forecast data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Historical and forecast dataframes.
    """
    df_hist = pd.DataFrame(columns=['ds', 'y', 'type'])
    df_forecast = pd.DataFrame(columns=['ds', 'yhat', 'type'])

    if os.path.exists(HISTORICAL_DATA_PATH):
        try:
            df_temp = pd.read_csv(HISTORICAL_DATA_PATH)
            df_temp.columns = df_temp.columns.str.lower().str.strip()
            
            if 'year' in df_temp.columns and 'week' in df_temp.columns:
                df_temp['ds'] = pd.to_datetime(
                    df_temp['year'].astype(str) + '-' + df_temp['week'].astype(str) + '-1',
                    format='%Y-%W-%w',
                    errors='coerce'
                )
            
            map_cols = {
                'val_cargaenergiahomwmed': 'y', 
                'carga': 'y', 
                'load': 'y'
            }
            df_temp.rename(columns=map_cols, inplace=True)

            if 'ds' in df_temp.columns and 'y' in df_temp.columns:
                df_temp = df_temp.dropna(subset=['ds'])
                df_temp['type'] = 'Historical'
                df_temp = df_temp.sort_values('ds')
                df_hist = df_temp[['ds', 'y', 'type']].copy()
            else:
                st.error(f"CSV Structure Error. Found columns: {list(df_temp.columns)}")

        except Exception as e:
            st.error(f"Error processing historical data: {e}")
    
    if os.path.exists(FORECAST_DATA_PATH):
        try:
            df_forecast = pd.read_csv(FORECAST_DATA_PATH)
            df_forecast['ds'] = pd.to_datetime(df_forecast['ds'])
            df_forecast['type'] = 'Forecast'
            
            if 'yhat_lower' not in df_forecast.columns: df_forecast['yhat_lower'] = df_forecast['yhat'] * 0.95
            if 'yhat_upper' not in df_forecast.columns: df_forecast['yhat_upper'] = df_forecast['yhat'] * 1.05
        except Exception as e:
            st.error(f"Error reading forecast file: {e}")

    return df_hist, df_forecast


def get_mlflow_metadata() -> mlflow.entities.Run | None:
    """Retrieve the latest MLflow run metadata for the Prophet model.
        Returns:
        mlflow.entities.Run | None: Latest run or None if not found.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient()
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="tags.model_type = 'Prophet'",
                order_by=["attribute.start_time DESC"],
                max_results=1
            )
            return runs[0] if runs else None
    except Exception:
        return None


def display_kpi_card(label, value, subtext=None, color_bar="#3FB950"):
    st.markdown(f"""
    <div class="metric-card-container" style="border-left: 4px solid {color_bar};">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-delta">{subtext if subtext else ''}</div>
    </div>
    """, unsafe_allow_html=True)

df_hist, df_forecast = load_combined_data()

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3103/3103446.png", width=50)
    st.title("Controls")
    st.markdown("---")
    
    if not df_hist.empty:
        min_date = df_hist['ds'].min().date()
        max_date = (df_forecast['ds'].max() if not df_forecast.empty else df_hist['ds'].max()).date()
        
        start_date = st.date_input("Start Date", value=max_date - timedelta(days=90), min_value=min_date, max_value=max_date)
        end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
    else:
        start_date, end_date = None, None

    st.markdown("### Visualization Layers")
    show_confidence = st.toggle("Show Confidence Interval (95%)", value=True)
    show_history = st.toggle("Show Historical Data", value=True)
    
    st.markdown("---")
    st.caption("v.2.1.0 • Powered by Prophet")

col_title, col_status = st.columns([3, 1])
with col_title:
    st.title("Demand Monitoring")
    st.markdown("#### Predictive Energy Intelligence")

latest_run = get_mlflow_metadata()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    model_ver = latest_run.data.tags.get("mlflow.runName", "N/A") if latest_run else "Offline"
    display_kpi_card("Model Version", model_ver, "Production", "#238636")

with kpi2:
    mae = f"{latest_run.data.metrics.get('training_mae', 0):.0f} MW" if latest_run else "0 MW"
    display_kpi_card("Mean Error (MAE)", mae, "Training Accuracy", "#A371F7")

with kpi3:
    rmse = f"{latest_run.data.metrics.get('training_rmse', 0):.0f} MW" if latest_run else "0 MW"
    display_kpi_card("RMSE", rmse, "Outlier Sensitivity", "#F778BA")

with kpi4:
    if not df_forecast.empty:
        avg_val = f"{df_forecast['yhat'].mean():.0f} MW"
        display_kpi_card("Forecasted Load", avg_val, "Period Average", "#29B5E8")
    else:
        display_kpi_card("Forecasted Load", "--", "No Data", "#29B5E8")

st.markdown("###") 

if df_hist.empty and df_forecast.empty:
    st.warning("Waiting for data pipeline...")
else:
    mask_hist = (df_hist['ds'].dt.date >= start_date) & (df_hist['ds'].dt.date <= end_date)
    df_hist_filtered = df_hist.loc[mask_hist]
    
    mask_forecast = (df_forecast['ds'].dt.date >= start_date)
    df_forecast_filtered = df_forecast.loc[mask_forecast]

    fig = go.Figure()

    if show_history and not df_hist_filtered.empty:
        fig.add_trace(go.Scatter(
            x=df_hist_filtered['ds'], 
            y=df_hist_filtered['y'],
            mode='lines',
            name='Actual History',
            line=dict(color='#424242', width=1.5),
            hovertemplate='%{y:.0f} MW<extra>History</extra>'
        ))

    if show_confidence and not df_forecast_filtered.empty:
        fig.add_trace(go.Scatter(
            x=pd.concat([df_forecast_filtered['ds'], df_forecast_filtered['ds'][::-1]]),
            y=pd.concat([df_forecast_filtered['yhat_upper'], df_forecast_filtered['yhat_lower'][::-1]]),
            fill='toself',
            fillcolor='rgba(41, 181, 232, 0.1)',
            line=dict(color='rgba(0,0,0,0)'),
            hoverinfo="skip",
            showlegend=False,
            name='Uncertainty'
        ))

    if not df_forecast_filtered.empty:
        fig.add_trace(go.Scatter(
            x=df_forecast_filtered['ds'], 
            y=df_forecast_filtered['yhat'],
            mode='lines+markers',
            name='AI Forecast',
            line=dict(color='#29B5E8', width=3),
            marker=dict(size=6, color='#29B5E8', symbol='circle'),
            hovertemplate='%{y:.0f} MW<extra>Forecast</extra>'
        ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E6EDF3', family="Inter, sans-serif"),
        height=550,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(
            showgrid=True, 
            gridcolor='#30363D',
            showline=True,
            linecolor='#30363D',
            rangeslider=dict(visible=True, thickness=0.05, bgcolor='#161B22')
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor='#30363D', 
            zeroline=False,
            title="Megawatts (MW)"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0)'
        ),
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("###")
    
    tab1, tab2 = st.tabs(["📋 Detailed Data", "📊 Drift Analysis"])
    
    with tab1:
        col_data, col_actions = st.columns([4, 1])
        with col_data:
            if not df_forecast.empty:
                display_df = df_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                display_df.columns = ['Date', 'Forecast (MW)', 'Lower Bound', 'Upper Bound']
                st.dataframe(
                    display_df.head(100), 
                    hide_index=True, 
                    use_container_width=True,
                    height=300
                )
            else:
                st.info("No forecast data available.")
        
        with col_actions:
            st.markdown("#### Actions")
            if not df_forecast.empty:
                csv = df_forecast.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name='forecast_export.csv',
                    mime='text/csv',
                    use_container_width=True
                )

    with tab2:
        st.info("Data Drift Detection module under development.")