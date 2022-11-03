import joblib
import json
import os
from google.cloud import storage
from google.oauth2 import service_account

from .model_parameters import ModelParameters

from zenml.client import Client
from zenml.steps import step, Output
from sklearn.pipeline import Pipeline

@step()
def deploy_model(
    deployment_decision: bool,
    model: Pipeline,
    parameters: ModelParameters
) -> Output(success=bool):
    """Writes the best model for inference.

    Args:
        deployment_decision (bool): Deployment decision indicates
                   whether the train model is better than previous 
                   runs.
        model (Pipeline): Model that predicts hotel bookings.
        parameters (ModelParameters): Parameters for loading the best
                          model to Google Store.

    Returns:
        success (bool): Sucess writing the model.
    """
    
    if deployment_decision:
        
        print("Storing a new model")
        absolute_path = os.path.dirname(__file__)
        full_path_service_account_file = os.path.join(absolute_path, 
                                                      "..", 
                                                      parameters.SERVICE_ACCOUNT_FILE)
        
        with open(full_path_service_account_file) as source:
            info = json.load(source)
        
        model_file_name = "model.pkl"
        
        try:
            print("Saving model locally")
            joblib.dump(
                model,
                parameters.MODEL_LOCAL_PATH + model_file_name)
            print("Saved best model locally.")
        except Exception as err:
            print("Error while saving best model: %s ", err)
            return False
        
        credentials = service_account.Credentials.from_service_account_info(info)
        storage_client = storage.Client(project=parameters.PROJECT_ID, credentials=credentials)

        
        try:
            print("Saving model to google cloud storage ")
            bucket = storage_client.bucket(parameters.GCP_BUCKET_NAME)
            blob = bucket.blob(parameters.GCP_BUCKET_MODEL_PATH + model_file_name)
            blob.upload_from_filename(parameters.MODEL_LOCAL_PATH + model_file_name)
            print('file: ', 
                  model_file_name,' uploaded to bucket: ',
                  parameters.GCP_BUCKET_NAME,' successfully')
        except Exception as err:
            print("Error while saving best model to google cloud storage: %s ", err)
            return False
            
        return True
    else:
        print("Best model is already stored.")
        return False