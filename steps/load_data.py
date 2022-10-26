import pandas as pd
from zenml.steps import step, Output


@step(enable_cache=False)
def load_training_data() -> Output(data=pd.DataFrame):

    """Loads the train data

    Returns:
        data: pd.DataFrame
    """
    try:
        print("Loading data")
        data = pd.read_csv("data/train_test_split/train.csv")
    except FileNotFoundError:
        print("CSV not found")
        data = pd.DataFrame()

    return data
