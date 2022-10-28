import pandas as pd
from zenml.steps import step, Output
from .model_parameters import ModelParameters


@step(enable_cache=False)
def load_training_data() -> Output(data=pd.DataFrame):

    """Loads the train data

    Returns:
        data: pd.DataFrame
    """
    try:
        print("Loading training data")
        data = pd.read_csv("data/train_test_split/train.csv")
    except FileNotFoundError:
        print("CSV not found")
        data = pd.DataFrame()

    return data


@step(enable_cache=False)
def load_inference_data(parameters: ModelParameters) -> Output(data=pd.DataFrame):

    """Loads the test data for inference

    Returns:
        data: pd.DataFrame
    """
    try:
        print("Loading inference data")
        data = pd.read_csv("data/train_test_split/test.csv")
    except FileNotFoundError:
        print("CSV not found")
        data = pd.DataFrame()

    return data