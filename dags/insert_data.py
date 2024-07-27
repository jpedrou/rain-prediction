import os
import sqlite3
import random
import string
import pandas as pd

from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator

from dotenv import load_dotenv

from datetime import datetime

load_dotenv()


DEFAULT_ARGS = {"owner": "airflow"}
DATABASE_PATH = os.getenv("DATABASE_PATH")

dag = DAG(
    dag_id="insert_data",
    default_args=DEFAULT_ARGS,
    schedule_interval='*/1 * * * *',
    start_date=days_ago(2),
    catchup=False,
)


def random_wind_direction():
    directions = [
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
    ]
    return random.choice(directions)


def random_rain_today():
    return random.choice(["Yes", "No"])


def generate_random_data():
    with sqlite3.connect(DATABASE_PATH, timeout=30) as conn:
        date = datetime.now().strftime('%Y-%m-%d')
        min_temp = round(random.uniform(5, 25), 1)
        max_temp = round(random.uniform(15, 35), 1)
        rainfall = round(random.uniform(0, 20), 1)
        evap = round(random.uniform(0, 10), 1)
        sun = round(random.uniform(0, 12), 1)
        windgustdir = random_wind_direction()
        windgustspeed = round(random.uniform(10, 60), 1)
        windd9 = random_wind_direction()
        windd3 = random_wind_direction()
        wins9 = round(random.uniform(0, 30), 1)
        wins3 = round(random.uniform(0, 30), 1)
        h9 = round(random.uniform(30, 100), 1)
        h3 = round(random.uniform(30, 100), 1)
        p9 = round(random.uniform(980, 1030), 1)
        p3 = round(random.uniform(980, 1030), 1)
        c9 = round(random.uniform(0, 8), 1)
        c3 = round(random.uniform(0, 8), 1)
        t9 = round(random.uniform(10, 30), 1)
        t3 = round(random.uniform(10, 30), 1)
        raintoday = random_rain_today()
        raintomorrow = random_rain_today()

        new_df = pd.DataFrame(
            {
                "Date": [date],
                "MinTemp": [min_temp],
                "MaxTemp": [max_temp],
                "Rainfall": [rainfall],
                "Evaporation": [evap],
                "Sunshine": [sun],
                "WindGustDir": [windgustdir],
                "WindGustSpeed": [windgustspeed],
                "WindDir9am": [windd9],
                "WindDir3pm": [windd3],
                "WindSpeed9am": [wins9],
                "WindSpeed3pm": [wins3],
                "Humidity9am": [h9],
                "Humidity3pm": [h3],
                "Pressure9am": [p9],
                "Pressure3pm": [p3],
                "Cloud9am": [c9],
                "Cloud3pm": [c3],
                "Temp9am": [t9],
                "Temp3pm": [t3],
                "RainToday": [raintoday],
                "RainTomorrow": [raintomorrow]
            }
        )

        new_df.to_sql('rain_in_australia', conn, if_exists='append', index=False)
        conn.commit()

generate_data = PythonOperator(task_id="extract_csv", python_callable=generate_random_data, dag=dag)