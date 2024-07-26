import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
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

pd.set_option("Display.max_columns", None)

# Load Data
df = pd.read_csv("../../data/processed/df_processed.csv")

df_new = df.copy()

# Split in X and y
X = df_new.drop(["RainTomorrow"], axis=1)
y = df_new["RainTomorrow"]

# Split into train and dev sets
X_train, X_dev, y_train, y_dev = train_test_split(
    X, y, stratify=y, test_size=0.015, random_state=0
)

# Preprocessing Pipeline
cat_features = X_train.drop('Date', axis = 1).select_dtypes(np.object_).columns
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
            MeanMedianImputer(variables=list(num_features), imputation_method="mean"),
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

# Preprocessing sets
X_train_processed = preprocess.fit_transform(X_train)
X_dev_processed = preprocess.transform(X_dev)


# Training Model

model = ExtraTreesClassifier(n_estimators=200, random_state=0, n_jobs=-1)

model.fit(X_train_processed, y_train)

# Results on train set
y_train_pred = model.predict(X_train_processed)
y_train_prob = model.predict_proba(X_train_processed)[:, 1]

print(f"AUC score on train set: {roc_auc_score(y_train, y_train_prob)}")
print("\n")
print("Classification Report on Train")
print(classification_report(y_train, y_train_pred))


# Results on dev set
y_dev_pred = model.predict(X_dev_processed)
y_dev_prob = model.predict_proba(X_dev_processed)[:, 1]

print(f"AUC score on dev set: {roc_auc_score(y_dev, y_dev_prob)}")
print("\n")
print("Classification Report on Dev")
print(classification_report(y_dev, y_dev_pred))


# Results on test set
test_set = pd.read_csv("../../data/processed/df_test_processed.csv")

# Split in X and y
X_test = test_set.drop(["RainTomorrow"], axis=1)
y_test = test_set["RainTomorrow"]

# Preprocessing test set
X_test_processed = preprocess.transform(X_test)

y_test_pred = model.predict(X_test_processed)
y_test_prob = model.predict_proba(X_test_processed)[:, 1]

print(f"AUC score on test set: {roc_auc_score(y_test, y_test_prob)}")
print("\n")
print("Classification Report on test")
print(classification_report(y_test, y_test_pred))
