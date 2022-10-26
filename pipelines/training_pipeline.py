from zenml.pipelines import pipeline


@pipeline(enable_cache=False)
def training_pipeline(
    load_training_data, clean_data, split_train_data, train_model, evaluate_model
):
    """Defines a training pipeline to train a model that
    predicts the cancellation of a hotel reservation.

    Args:
        load_train_data: Loads data for training
        clean_data: Cleans the training data
        split_train_data: Splits the train data
        train_model: Trains the model
        evaluate_model: Evaluates the model
    """

    data = load_training_data()
    cleaned_data = clean_data(data)
    X_train, X_valid, y_train, y_valid = split_train_data(cleaned_data)
    model = train_model(X_train, y_train)
    f1_sc = evaluate_model(model, X_valid, y_valid)

    print(f1_sc)
