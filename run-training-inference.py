from steps.load_data import load_training_data
from steps.clean_data import clean_data
from steps.split_train_data import split_train_data
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model
from steps.evaluate_deployment import evaluate_deployment
from steps.deploy_model import deploy_model
from steps.load_data import load_inference_data
from steps.prediction_steps import prediction_service_loader
from steps.prediction_steps import predictor
from steps.drift_detection import drift_detector
from pipelines.training_pipeline import training_pipeline
from pipelines.inference_pipeline import inference_pipeline

from zenml.integrations.evidently.visualizers import EvidentlyVisualizer

def pipeline_run():
    """Runs training pipeline
    """
    training_pipeline(
        load_training_data=load_training_data(),
        clean_data=clean_data(),
        split_train_data=split_train_data(),
        train_model=train_model(),
        evaluate_model=evaluate_model(),
        evaluate_deployment=evaluate_deployment(),
        deploy_model=deploy_model,
    ).run()
    
def inference_run():
    """Runs inference pipeline
    """
    
    inference_pipeline(
        inference_data_loader=load_inference_data(),
        prediction_service_loader=prediction_service_loader(),
        predictor=predictor(),
        training_data_loader=load_training_data(),
        drift_detector=drift_detector,
    )
    
    def visualize_data_drift():
        """Visualize the data drift
        """
        inf_run = inference_pipeline_instance.get_runs()[-1]
        drift_detection_step = inf_run.get_step(step="drift_detector")
        EvidentlyVisualizer().visualize(drift_detection_step)

if __name__ == "__main__":
    pipeline_run()
    inference_run()
    visualize_data_drift()
