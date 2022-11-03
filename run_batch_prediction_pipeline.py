from steps.load_data import load_inference_data
from steps.clean_data import clean_data_inference
from steps.fetch_model import fetch_model
from steps.predict import get_predictions
from steps.store_predictions import store_predictions

from steps.prediction_steps import prediction_service_loader
from steps.prediction_steps import predictor
from steps.drift_detection import drift_detector
from pipelines.batch_inference_pipeline import batch_inference_pipeline

#from zenml.integrations.evidently.visualizers import EvidentlyVisualizer


def batch_inference_run():
    """Runs inference pipeline
    """

    batch_inference_pipeline_instance = batch_inference_pipeline(
        load_inference_data=load_inference_data(),
        clean_data_inference=clean_data_inference(),
        fetch_model=fetch_model(),
        get_predictions=get_predictions(),
        store_predictions=store_predictions(),
        #prediction_service_loader=prediction_service_loader(),
        #predictor=predictor(),
        #load_training_data=load_training_data(),
        #drift_detector=drift_detector,
    )

    batch_inference_pipeline_instance.run()

    #inf_run = inference_pipeline_instance.get_runs()[-1]
    #drift_detection_step = inf_run.get_step(step="drift_detector")
    #EvidentlyVisualizer().visualize(drift_detection_step)


if __name__ == "__main__":
    batch_inference_run()
