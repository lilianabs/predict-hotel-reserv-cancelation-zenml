import pandas as pd
from .model_parameters import ModelParameters
from sklearn.model_selection import train_test_split
from zenml.steps import step, Output

@step
def split_train_data(cleaned_data: pd.DataFrame,
                     parameters: ModelParameters) -> Output(
    X_train=pd.DataFrame,
    X_valid=pd.DataFrame,
    y_train=pd.Series,
    y_valid=pd.Series
):
    """Splits the train data for the model.

    Args:
        cleaned_data (pd.DataFrame): Train data for splitting.
        paramters (ModelParameters): Features to select.

    Returns:
        X_train (pd.DataFrame): Features for training the model.
        X_valid (pd.DataFrame): Features for evaluating the model.
        y_train (pd.Series):  Target for training the model.
        y_valid (pd.Series): Target for evaluating the model.
    """
    
    X = cleaned_data[parameters.NUM_FEATURES + parameters.CAT_FEATURES]
    y = cleaned_data.is_canceled
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=42)
    
    return X_train, X_valid, y_train, y_valid