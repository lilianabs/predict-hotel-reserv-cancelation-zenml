import mlflow
import pandas as pd

from sklearn.metrics import accuracy_score

from zenml.client import Client
from zenml.steps import step, Output
from sklearn.pipeline import Pipeline

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    model: Pipeline, X_valid: pd.DataFrame, y_valid: pd.Series
) -> Output(acc=float):
    """Evaluates the model using test data.

    Args:
        model (Pipeline): Model that predicts hotel bookings.
        X_valid (pd.DataFrame): Features for predicting.
        y_valid (pd.DataFrame): Target for predicting.

    Returns:
        acc (float): Accuracy of the model.
    """

    y_predict = model.predict(X_valid)
    test_acc = accuracy_score(y_valid, y_predict)
    mlflow.log_metric("test_accuracy", test_acc)

    return test_acc
