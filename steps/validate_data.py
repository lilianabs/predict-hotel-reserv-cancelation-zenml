from .model_parameters import ModelParameters

from zenml.integrations.deepchecks.steps import (
    DeepchecksDataIntegrityCheckStepParameters,
    deepchecks_data_integrity_check_step,
)

params = ModelParameters()


validate_data = deepchecks_data_integrity_check_step(
    step_name="validate_data",
    params=DeepchecksDataIntegrityCheckStepParameters(
        dataset_kwargs=dict(label=params.LABEL, 
                            cat_features=params.CAT_FEATURES),
    ),
)