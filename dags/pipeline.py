import pandas as pd
import numpy as np
import sqlite3
import pickle
import git
import os

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import ExtraTreesClassifier

from feature_engine.selection import (
    DropConstantFeatures,
    DropCorrelatedFeatures,
    DropDuplicateFeatures,
)
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.encoding import OrdinalEncoder
from feature_engine.datetime import DatetimeFeatures
from feature_engine.outliers import Winsorizer

from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator

from dotenv import load_dotenv

load_dotenv()

DEFAULT_ARGS = {"owner": "airflow"}
ROOT_PATH = os.getenv("ROOT_PATH")
IMAGES_PATH = os.getenv("IMAGES_PATH")
SAVE_DATA_PATH = os.getenv("SAVE_DATA_PATH")
DATABASE_PATH = os.getenv("DATABASE_PATH")
TOKEN = os.getenv("TOKEN")

dag = DAG(
    dag_id="pipeline",
    default_args=DEFAULT_ARGS,
    schedule_interval="@daily",
    start_date=days_ago(2),
    catchup=False,
)


def extract_csv():

    query = """SELECT * FROM rain_in_australia"""
    connection = sqlite3.connect(DATABASE_PATH)

    df = pd.read_sql(query, con=connection)
    df.to_csv(os.path.join(SAVE_DATA_PATH, "data.csv"), index=None)

    connection.close()


def preprocess_data():

    df = pd.read_csv(os.path.join(SAVE_DATA_PATH, "data.csv"))
    X = df.drop(["RainTomorrow", "index"], axis=1)
    y = df["RainTomorrow"]

    X_train, X_dev, y_train, y_dev = train_test_split(
        X, y, stratify=y, test_size=0.015, random_state=0
    )

    cat_features = X_train.drop("Date", axis=1).select_dtypes(np.object_).columns
    num_features = X_train.select_dtypes(np.number).columns

    preprocess = Pipeline(
        [
            (
                "Object Imputation",
                CategoricalImputer(
                    imputation_method="frequent",
                    variables=list(cat_features),
                ),
            ),
            (
                "Get Data Features",
                DatetimeFeatures(
                    variables="Date",
                    drop_original=True,
                    features_to_extract=["month", "year", "day_of_month"],
                ),
            ),
            (
                "Numerical Imputation",
                MeanMedianImputer(
                    variables=list(num_features), imputation_method="mean"
                ),
            ),
            ("Handle Outliers", Winsorizer(variables=list(num_features))),
            (
                "Object Enconding",
                OrdinalEncoder(
                    variables=list(cat_features), encoding_method="arbitrary"
                ),
            ),
            ("Drop Constant Features", DropConstantFeatures()),
            ("Drop Duplicated Features", DropDuplicateFeatures()),
            ("Drop High Correlated Features", DropCorrelatedFeatures(threshold=0.8)),
        ]
    )

    X_train_processed = preprocess.fit_transform(X_train)
    X_dev_processed = preprocess.transform(X_dev)

    with open(os.path.join(ROOT_PATH, "preprocess_pipe.pkl"), "wb") as f:
        pickle.dump(preprocess, f)

    with open(os.path.join(SAVE_DATA_PATH, "training_sets.pkl"), "wb") as f:
        pickle.dump(X_train_processed, f)
        pickle.dump(X_dev_processed, f)
        pickle.dump(y_train, f)
        pickle.dump(y_dev, f)


def train_model():

    def plot_roc_curve(true_y, y_prob, auc_score):
        """
        plots the roc curve based of the probabilities
        """

        fpr, tpr, _ = roc_curve(true_y, y_prob)
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend((f"AUC Score: {auc_score}",), loc="best")
        plt.savefig(os.path.join(IMAGES_PATH, "roc_curve.png"), transparent=False)
        plt.show()

    with open(os.path.join(SAVE_DATA_PATH, "training_sets.pkl"), "rb") as f:
        X_train_processed = pickle.load(f)
        X_dev_processed = pickle.load(f)
        y_train = pickle.load(f)
        y_dev = pickle.load(f)

    model = ExtraTreesClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    model.fit(X_train_processed, y_train)

    y_dev_prob = model.predict_proba(X_dev_processed)[:, 1]

    auc_score = round(roc_auc_score(y_dev, y_dev_prob), 3) * 100

    plot_roc_curve(y_dev.map({"No": 0, "Yes": 1}), y_dev_prob, auc_score)

    with open(os.path.join(ROOT_PATH, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    # Update Commits
    repo = git.Repo(ROOT_PATH)
    changed_files = [item.a_path for item in repo.index.diff(None)]
    untracked_files = repo.untracked_files

    repo.index.add(changed_files + untracked_files)
    repo.index.commit('New Repo Update')

    origin = repo.remote(name = 'origin')
    origin.set_url(f'https://{TOKEN}@github.com/jpedrou/rain-prediction.git')
    origin.push()
    


# Dags Operations
extract = PythonOperator(task_id="extract_csv", python_callable=extract_csv, dag=dag)
preprocess = PythonOperator(
    task_id="preprocess_data", python_callable=preprocess_data, dag=dag
)
train = PythonOperator(task_id="train_model", python_callable=train_model, dag=dag)

# Queue Sequence
extract >> preprocess >> train
