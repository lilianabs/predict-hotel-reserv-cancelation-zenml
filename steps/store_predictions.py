import pandas as pd
import numpy as np
import json
import os
from zenml.steps import step, Output

from .model_parameters import ModelParameters
from google.cloud import storage
from google.oauth2 import service_account

@step
def store_predictions(predictions: np.ndarray,
                      parameters: ModelParameters) -> Output():
    
    print("Preparing predictions data")
    data = pd.DataFrame(data=predictions, columns=["pred_is_canceled"])
    predictions_file_name= "predictions.csv" 
    data.to_csv(parameters.MODEL_LOCAL_PATH + predictions_file_name)
    
    absolute_path = os.path.dirname(__file__)
    full_path_service_account_file = os.path.join(absolute_path, 
                                                  "..", 
                                                  parameters.SERVICE_ACCOUNT_FILE)
        
    with open(full_path_service_account_file) as source:
        info = json.load(source)
        
    credentials = service_account.Credentials.from_service_account_info(info)
    storage_client = storage.Client(project=parameters.PROJECT_ID, credentials=credentials)
    
    try:
        print("Saving predictions to google cloud storage ")
        bucket = storage_client.bucket(parameters.GCP_BUCKET_NAME)
        blob = bucket.blob(parameters.GCP_BUCKET_PREDS_FILE_PATH)
        blob.upload_from_filename(parameters.MODEL_LOCAL_PATH + predictions_file_name)
        print('file: ', 
              predictions_file_name,' uploaded to bucket: ',
              parameters.GCP_BUCKET_NAME,' successfully')
    except Exception as err:
        print("Error while storing predictions to google cloud storage: ", err)
