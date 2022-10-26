import mlflow
import pandas as pd
from .model_parameters import ModelParameters

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

from zenml.client import Client
from zenml.steps import step, Output

experiment_tracker = Client().active_stack.experiment_tracker


@step(enable_cache=False, experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame, y_train: pd.Series, parameters: ModelParameters
) -> Output(model=Pipeline):

    num_transformer = SimpleImputer(strategy="constant")

    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, parameters.NUM_FEATURES),
            ("cat", cat_transformer, parameters.CAT_FEATURES),
        ]
    )

    params = {"n_estimators": 10, "max_depth": 5}

    mlflow.log_params(params)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(**params, random_state=42)),
        ]
    )

    model.fit(X_train, y_train)

    y_predict = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_predict)
    mlflow.log_metric("train_accuracy", train_acc)

    return model
