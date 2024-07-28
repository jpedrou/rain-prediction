<p align="center"><img width="150" src="images/kangaroo.png"/></p>
<h1 align="center">Rain Prediction in Australia</h1>

**Objective**

Develop a system that predicts if it will rain today and tomorrow in Australia using machine learning. Additionally, implement an user interface and a queue system using Apache Airflow to continuously update the model whenever the dataset changes.

**Technologies** 

Python|Streamlit|Airflow|SQLite
---|----|----|----|
<img width="40" src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/python/python-original.svg" />|<img width="40" src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/streamlit/streamlit-original.svg" />|<img width="40" src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/apacheairflow/apacheairflow-original.svg" />|<img width="40" src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/sqlite/sqlite-original.svg" />

## Project Components

1. **Data Collection and Preparation**
    - Use the weather dataset provided by Kaggle: [Rain in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package).
    - Perform Exploratory Data Analysis (EDA).
    - Preprocess the data to handle missing values, encode categorical variables and more.

2. **Modeling**
    - Select appropriate machine learning algorithm.
    - Train and validate the models using historical weather data to find the best performing model.

3. **Prediction System**
    - Develop a Python application that takes daily weather features as input and predicts whether it will rain today and tomorrow.
    - Implement the prediction logic using the trained machine learning model.


4. **Queue System with Apache Airflow**
    - Set up Apache Airflow to automate the workflow of updating the model.
    - Create DAGs (Directed Acyclic Graphs) in Airflow to define tasks such as data ingestion, preprocessing, model training, and deployment.
    - Schedule the DAGs to run at regular intervals or when new data is available.
