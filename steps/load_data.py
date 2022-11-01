import pandas as pd
import os
from zenml.steps import step, Output
from .model_parameters import ModelParameters


@step(enable_cache=False)
def load_training_data(parameters: ModelParameters) -> Output(data=pd.DataFrame):

    """Loads the train data

    Returns:
        data: pd.DataFrame
    """
    
    try:
        print("Loading training data")
        absolute_path = os.path.dirname(__file__)
        full_path_service_account_file = os.path.join(absolute_path, 
                                                      "..", 
                                                      parameters.SERVICE_ACCOUNT_FILE)
        data = pd.read_csv(parameters.TRAIN_DATA_PATH, 
                           storage_options={"token": full_path_service_account_file})
    except FileNotFoundError:
        print("CSV not found")
        data = pd.DataFrame()

    return data


@step(enable_cache=False)
def load_test_data(parameters: ModelParameters) -> Output(data=pd.DataFrame):

    """Loads the test data for inference

    Returns:
        data: pd.DataFrame
    """
    try:
        print("Loading inference data")
        absolute_path = os.path.dirname(__file__)
        full_path = os.path.join(absolute_path, "..", parameters.SERVICE_ACCOUNT_FILE)
        data = pd.read_csv(parameters.TEST_DATA_PATH, 
                           storage_options={"token": full_path})
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
        absolute_path = os.path.dirname(__file__)
        full_path = os.path.join(absolute_path, "..", parameters.SERVICE_ACCOUNT_FILE)
        data = pd.read_csv(parameters.INFERENCE_DATA_PATH, 
                           storage_options={"token": full_path})
    except FileNotFoundError:
        print("CSV not found")
        data = pd.DataFrame()

    return data