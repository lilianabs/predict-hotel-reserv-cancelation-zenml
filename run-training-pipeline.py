import pandas as pd
import numpy as np
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# zenml importing
from zenml.steps import step, Output
from zenml.pipelines import pipeline
from zenml.client import Client

NUM_FEATURES = [
        'lead_time', 'arrival_date_week_number', "arrival_date_day_of_month",
        'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children',
        'babies', 'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled',
        'required_car_parking_spaces', 'total_of_special_requests', 'adr'
]

CAT_FEATURES = [
        'hotel', 'agent', 'arrival_date_month', 'meal', 'market_segment',
        'distribution_channel', 'reserved_room_type', 'deposit_type', 'customer_type'
]

experiment_tracker = Client().active_stack.experiment_tracker


@step(enable_cache=False)
def load_data() -> Output(
    data=pd.DataFrame
):
    
    """Load the training data"""
    try:
        print("Loading data")
        data = pd.read_csv("data/raw/hotel_booking.csv")
    except FileNotFoundError:
        print("CSV not found")
        data = pd.DataFrame()

    return data

@step(enable_cache=False)
def clean_data(data: pd.DataFrame) -> Output(
    cleaned_data=pd.DataFrame
):

    # Select features
    cleaned_data = data[NUM_FEATURES + CAT_FEATURES + ['is_canceled']].copy()
    
    # Convert all cat features that are float into int
    for col in CAT_FEATURES:
        if cleaned_data[col].dtype == 'float':
            cleaned_data[col] = cleaned_data[col].fillna(0).astype(int)
    
    return cleaned_data

@step
def split_train_data(cleaned_data: pd.DataFrame) -> Output(
    X_train=pd.DataFrame,
    X_valid=pd.DataFrame,
    y_train=pd.Series,
    y_valid=pd.Series
):
    
    X = cleaned_data[NUM_FEATURES + CAT_FEATURES]
    y = cleaned_data.is_canceled
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=42)
    
    return X_train, X_valid, y_train, y_valid


@step(enable_cache=False, experiment_tracker=experiment_tracker.name)
def train_model(X_train: pd.DataFrame,
                y_train: pd.Series,
                ) -> Output(
    model=Pipeline
):
                    
    #mlflow.sklearn.autolog()
    
    num_transformer = SimpleImputer(strategy='constant')

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, NUM_FEATURES),
        ('cat', cat_transformer, CAT_FEATURES)
    ])
    
    params = {
        "n_estimators": 10,
        "max_depth": 5
    }
    
    mlflow.log_params(params)
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(**params, random_state=42))
    ])
    
    model.fit(X_train, y_train)
    
    y_predict = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_predict)
    mlflow.log_metric("train_accuracy", train_acc)
    
    return model

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: Pipeline,
                   X_valid: pd.DataFrame, y_valid: pd.Series) -> Output(
                       acc=float
                   ):
    
    
    y_predict = model.predict(X_valid)
    test_acc = accuracy_score(y_valid, y_predict)
    mlflow.log_metric("test_accuracy", test_acc)
    
    return test_acc


@pipeline(enable_cache=False)
def training_pipeline(
    load_data,
    clean_data,
    split_train_data,
    train_model,
    evaluate_model
):
    
    data = load_data()
    cleaned_data = clean_data(data)
    X_train, X_valid, y_train, y_valid = split_train_data(cleaned_data)
    model = train_model(X_train, y_train)
    f1_sc = evaluate_model(model, X_valid, y_valid)
    
    print(f1_sc)


training_pipeline(
    load_data=load_data(),
    clean_data=clean_data(),
    split_train_data=split_train_data(),
    train_model=train_model(),
    evaluate_model=evaluate_model()
).run()





