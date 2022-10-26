import pandas as pd
from .model_parameters import ModelParameters
from zenml.steps import step, Output


@step(enable_cache=False)
def clean_data(
    data: pd.DataFrame, parameters: ModelParameters
) -> Output(cleaned_data=pd.DataFrame):
    """Cleans the data for training and inference:
           Selects features
           Converts float categorical features to int

    Args:
        data (pd.DataFrame): Data for cleaning. Defaults to pd.DataFrame.
        parameters (ModelParameters): Features to select.

    Returns:
        _type_: _description_
    """

    # Select features
    cleaned_data = data[
        parameters.NUM_FEATURES + parameters.CAT_FEATURES + ["is_canceled"]
    ].copy()

    # Convert all cat features that are float into int
    for col in parameters.CAT_FEATURES:
        if cleaned_data[col].dtype == "float":
            cleaned_data[col] = cleaned_data[col].fillna(0).astype(int)

    return cleaned_data
