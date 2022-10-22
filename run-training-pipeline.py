import pandas as pd
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn. compose import ColumnTransformer

from category_encoders import CountEncoder

# zenml importing
from zenml.steps import step, Output
from zenml.pipelines import pipeline
from zenml.client import Client

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
def perform_feature_preprocessing(data: pd.DataFrame) -> Output(
    data_preprocessed=pd.DataFrame
):

     # Extract year, month, and date from the 
     # reservation_status_column
    data['reservation_status_date'] = pd.to_datetime(data['reservation_status_date'])
    data['reservation_status_date_year'] = data['reservation_status_date'].dt.year
    data['reservation_status_date_month'] = data['reservation_status_date'].dt.month
    data['reservation_status_date_day'] = data['reservation_status_date'].dt.day
    data = data.drop('reservation_status_date', axis=1)
    
    # Drop features
    features_to_drop = ['name', 'email', 'phone-number', 'credit_card', 'company']
    data = data.drop(features_to_drop, axis=1)
    
    data_preprocessed = data.copy()
        
    return data_preprocessed

@step
def split_train_data(data: pd.DataFrame) -> Output(
    X_train=pd.DataFrame,
    X_test=pd.DataFrame,
    y_train=pd.Series,
    y_test=pd.Series
):
    
    X = data.drop('is_canceled', axis=1)
    y = data.is_canceled
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y,
                                                        random_state=0,
                                                        train_size=0.8)
    
    return X_train, X_test, y_train, y_test


@step(enable_cache=False, experiment_tracker=experiment_tracker.name)
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Output(
    model_pipeline=Pipeline
):
    mlflow.sklearn.autolog()
    
    features_to_impute = ['children', 'agent']
    features_to_onehot_encode = ['hotel', 'arrival_date_month', 
                                 'meal', 'deposit_type', 'customer_type', 
                                 'reservation_status']
    features_to_count_encode = ['country', 'market_segment', 
                                'distribution_channel', 'reserved_room_type', 
                                'assigned_room_type']
    
    imputer = SimpleImputer(fill_value=0)
    onehot_enc = OneHotEncoder()
    count_enc = CountEncoder()
    
    preprocessor = ColumnTransformer(
    transformers=[('imputer', imputer, features_to_impute),
            ('onehot', onehot_enc, features_to_onehot_encode),
            ('count_encode', count_enc, features_to_count_encode)
    ])
    
    model = LogisticRegression()
    
    model_pipeline = Pipeline(
    steps=[
       ('preprocessor', preprocessor),
       ('model', model)
    ])
    
    model_pipeline.fit(X_train, y_train)
    
    return model_pipeline


@step(enable_cache=False)
def evaluate_model(model_pipeline: Pipeline,
                   X_test: pd.DataFrame, y_test: pd.Series) -> Output(
                       f1=float
                   ):
    
    y_predict = model_pipeline.predict(X_test)
    
    f1_sc = f1_score(y_test, y_predict)
    
    return f1_sc


@pipeline(enable_cache=False)
def training_pipeline(
    load_data,
    perform_feature_preprocessing,
    split_train_data,
    train_model,
    evaluate_model
):
    
    data = load_data()
    data_preprocessed = perform_feature_preprocessing(data)
    X_train, X_test, y_train, y_test = split_train_data(data_preprocessed)
    model = train_model(X_train, y_train)
    f1_sc = evaluate_model(model, X_test, y_test)
    
    print(f1_sc)


training_pipeline(
    load_data=load_data(),
    perform_feature_preprocessing=perform_feature_preprocessing(),
    split_train_data=split_train_data(),
    train_model=train_model(),
    evaluate_model=evaluate_model()
).run()





