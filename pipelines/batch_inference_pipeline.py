from steps.clean_data import clean_data_inference
from zenml.pipelines import pipeline


@pipeline
def batch_inference_pipeline(
    load_inference_data,
    clean_data_inference,
    fetch_model,
    get_predictions,
    store_predictions
    # prediction_service_loader,
    # predictor,
    # load_training_data,
    # drift_detector,
):
    """Inference pipeline with skew and drift detection."""
    data = load_inference_data()
    cleaned_data = clean_data_inference(data=data)
    model = fetch_model()
    predictions = get_predictions(model, cleaned_data)
    store_predictions(predictions)

    # model_deployment_service = prediction_service_loader()
    # predictor(model_deployment_service, inference_data)
    # training_data = load_training_data()
    # drift_detector(training_data, inference_data)
