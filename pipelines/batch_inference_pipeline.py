from zenml.pipelines import pipeline


@pipeline
def batch_inference_pipeline(
    load_inference_data,
    clean_data_inference,
    fetch_model,
    get_predictions,
    store_predictions

):
    """Inference pipeline with skew and drift detection."""
    data = load_inference_data()
    cleaned_data = clean_data_inference(data=data)
    model = fetch_model()
    predictions = get_predictions(model, cleaned_data)
    store_predictions(predictions)

