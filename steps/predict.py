import pandas as pd
import numpy as np
from zenml.steps import step, Output
from sklearn.pipeline import Pipeline


@step
def get_predictions(
    model: Pipeline, cleaned_data: pd.DataFrame
) -> Output(predictions=np.ndarray):

    """Gets predictions.
    
    Args:
        model (Pipeline): Best model.
        data (pd.DataFrame): Features for computing predictions.

    Returns:
        predictions (pd.Series): Predictions.
    """

    predictions = model.predict(cleaned_data)

    return predictions
