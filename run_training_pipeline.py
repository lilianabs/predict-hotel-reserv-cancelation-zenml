from steps.load_data import load_training_data
from steps.clean_data import clean_data
from steps.split_train_data import split_train_data
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model
from steps.evaluate_deployment import evaluate_deployment
from steps.deploy_model import deploy_model
from pipelines.training_pipeline import training_pipeline



def pipeline_run():
    """Runs training pipeline
    """
    training_pipeline_instance = training_pipeline(
        load_training_data=load_training_data(),
        clean_data=clean_data(),
        split_train_data=split_train_data(),
        train_model=train_model(),
        evaluate_model=evaluate_model(),
        evaluate_deployment=evaluate_deployment(),
        deploy_model=deploy_model(),
    )

    training_pipeline_instance.run()


if __name__ == "__main__":
    pipeline_run()