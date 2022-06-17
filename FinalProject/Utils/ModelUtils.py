import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline


def train_pipeline(X_train, y_train):
    pipeline_steps = [
        ('imputation', SimpleImputer()),
        ('normalization', StandardScaler()),
        ('model', DecisionTreeClassifier()),
    ]

    pipeline = Pipeline(pipeline_steps)

    pipeline.fit(X_train, y_train)

    return pipeline

