from steps.load_data import load_inference_data
from steps.clean_data import clean_data_inference
from steps.fetch_model import fetch_model
from steps.predict import get_predictions
from steps.store_predictions import store_predictions
from pipelines.batch_inference_pipeline import batch_inference_pipeline



def batch_inference_run():
    """Runs inference pipeline
    """

    batch_inference_pipeline_instance = batch_inference_pipeline(
        load_inference_data=load_inference_data(),
        clean_data_inference=clean_data_inference(),
        fetch_model=fetch_model(),
        get_predictions=get_predictions(),
        store_predictions=store_predictions(),
    )

    batch_inference_pipeline_instance.run()


if __name__ == "__main__":
    batch_inference_run()
