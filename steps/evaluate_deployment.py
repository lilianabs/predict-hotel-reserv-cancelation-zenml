from zenml.steps import step, Output

@step
def evaluate_deployment(test_acc: float) -> bool:
    """Only deploy if the test accuracy > 70%."""
    return test_acc > 0.9