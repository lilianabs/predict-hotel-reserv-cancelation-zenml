import json
import os 
import joblib

from zenml.steps import step, Output
from .model_parameters import ModelParameters
from io import BytesIO
from sklearn.pipeline import Pipeline

from google.cloud import storage
from google.oauth2 import service_account

@step(enable_cache=False)
def fetch_model(parameters: ModelParameters) -> Output(model=Pipeline):
    
    """Fetches the best model from the GCP bucket
    
    Args:
        parameters (ModelParameters): Parameters for loading the best
                          model to Google Store.

    Returns:
        model (Pipeline): Best model stored in GCP.
    """


    absolute_path = os.path.dirname(__file__)
    full_path_service_account_file = os.path.join(absolute_path, 
                                                  "..", 
                                                  parameters.SERVICE_ACCOUNT_FILE)
        
    with open(full_path_service_account_file) as source:
        info = json.load(source)
        
    print("Fetching latest model from GCP bucket.")
    credentials = service_account.Credentials.from_service_account_info(info)
    storage_client = storage.Client(project=parameters.PROJECT_ID, credentials=credentials)
    
    bucket = storage_client.get_bucket(parameters.GCP_BUCKET_NAME)
    blob = bucket.get_blob(parameters.GCP_BUCKET_MODEL_FILE_PATH)
    
    model = joblib.load(BytesIO(blob.download_as_string()))
    
    return model