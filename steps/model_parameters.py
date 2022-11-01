import os
from zenml.steps import step, BaseParameters


class ModelParameters(BaseParameters):
    """Parameters for the hotel reservation model

    Attributes:
        CAT_FEATURES (list): Categorical features for
              training the model.
        NUM_FEATURES (list): Numerical features for
              training the model.
    """

    CAT_FEATURES: list[str] = [
        "hotel",
        "agent",
        "arrival_date_month",
        "meal",
        "market_segment",
        "distribution_channel",
        "reserved_room_type",
        "deposit_type",
        "customer_type",
    ]

    NUM_FEATURES: list[str] = [
        "lead_time",
        "arrival_date_week_number",
        "arrival_date_day_of_month",
        "stays_in_weekend_nights",
        "stays_in_week_nights",
        "adults",
        "children",
        "babies",
        "is_repeated_guest",
        "previous_cancellations",
        "previous_bookings_not_canceled",
        "required_car_parking_spaces",
        "total_of_special_requests",
        "adr",
    ]
    
    SERVICE_ACCOUNT_FILE = "credentials.json"
    TRAIN_DATA_PATH = "gs://hotel-booking-prediction/data/train/train.csv"
    TEST_DATA_PATH = "gs://hotel-booking-prediction/data/test/test.csv"
    PROJECT_ID = "direct-cocoa-365217"
    MODEL_LOCAL_PATH = "models/"
    GCP_BUCKET_NAME = "hotel-booking-prediction"
    GCP_BUCKET_MODEL_PATH = "data/models/"