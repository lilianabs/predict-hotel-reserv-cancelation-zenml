from zenml.pipelines import pipeline

@pipeline
def batch_inference_pipeline(
    load_inference_data,
    prediction_service_loader,
    predictor,
    load_training_data,
    drift_detector,
):
    """Inference pipeline with skew and drift detection."""
    inference_data = load_inference_data()
    model_deployment_service = prediction_service_loader()
    predictor(model_deployment_service, inference_data)
    training_data = load_training_data()
    drift_detector(training_data, inference_data)

