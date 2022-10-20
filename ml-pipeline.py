# %%
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# zenml importing
from zenml.steps import step, Output
from zenml.pipelines import pipeline

# %%
@step
def load_data() -> Output(
    data=pd.DataFrame
):
    
    """Load the training data"""

    try:
        data = pd.read_csv("data/raw/fraudTrain.csv")
        data = data.drop('Unnamed: 0', axis=1)
    except FileNotFoundError:
        data = pd.DataFrame()

    return data

# %%
@step
def perform_feature_engineering(data: pd.DataFrame) -> Output(
    data_transfom=pd.DataFrame
):
    data_transform = data.copy()
    # Get transaction hour
    data_transform["trans_date_trans_time"] = pd.to_datetime(data_transform["trans_date_trans_time"])
    data_transform["hour"] = data_transform.trans_date_trans_time.dt.hour

    # Normal hours are between 05:00 and 21:00 and abnormal otherwise
    data_transform["is_normal_hour"] = 0
    data_transform.loc[data_transform.hour < 5, "is_normal_hour"] = 1
    data_transform.loc[data_transform.hour > 21, "is_normal_hour"] = 1
    
    return data_transform

# %%
@step
def split_train_data(data_transform: pd.DataFrame) -> Output(
    X_train=pd.DataFrame,
    X_test=pd.DataFrame,
    y_train=pd.Series,
    y_test=pd.Series
):
    features = ['amt', 'is_normal_hour']
    
    X = data_transform[features]
    y = data_transform.is_fraud
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y,
                                                        random_state=0,
                                                        train_size=0.8)
    
    return X_train, X_test, y_train, y_test

# %%
@step(enable_cache=False)
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Output(
    model=RandomForestClassifier
):
    
    params = {
        "n_estimators": 10,
        "max_depth": 5
    }
    
    model = RandomForestClassifier(
        **params,
        random_state=1
        )
    
    model.fit(X_train, y_train)
    
    return model

# %%
@step(enable_cache=False)
def evaluate_model(model: RandomForestClassifier,
                   X_test: pd.DataFrame, y_test: pd.Series) -> Output(
                       f1=float
                   ):
    
    y_predict = model.predict(X_test)
    
    f1_sc = f1_score(y_test, y_predict)
    
    return f1_sc

# %%
@pipeline(enable_cache=False)
def training_pipeline(
    load_data,
    perform_feature_engineering,
    split_train_data,
    train_model,
    evaluate_model
):
    
    data = load_data()
    data_transform = perform_feature_engineering(data)
    X_train, X_test, y_train, y_test = split_train_data(data_transform)
    model = train_model(X_train, y_train)
    f1_sc = evaluate_model(model, X_test, y_test)
    
    print(f1_sc)

# %%
training_pipeline(
    load_data=load_data(),
    perform_feature_engineering=perform_feature_engineering(),
    split_train_data=split_train_data(),
    train_model=train_model(),
    evaluate_model=evaluate_model()
).run()

# %%



