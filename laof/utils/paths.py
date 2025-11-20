from pathlib import Path

import os

MAX_DATA_CHUNKS = 80

storage_path = Path.cwd()
current_storage_path = Path(".")
_expert_data_path = storage_path / "expert_data" / "default"
_expert_datarandom_path = storage_path / "expert_data" / "random"
_expert_datamixed_path = storage_path / "expert_data" / "mixed"
_expert_datamixed_20_80_path = storage_path / "expert_data" / "mixed_20_80"
_expert_datamixed_40_80_path = storage_path / "expert_data" / "mixed_40_80"
_expert_dataopticalflow_path = storage_path / "expert_data" / "opticalflow"
_experiment_results_path = storage_path / "exp_results"
_current_experiment_results_path = current_storage_path / "exp_results"

assert (
    _expert_data_path.exists()
), f"Expert data dir: {_expert_data_path} does not exist"


def get_expert_data(env_name: str, data_type: str, test: bool) -> list[Path]:
    test_flag = "test" if test else "train"
    if data_type == "default":
        task_data_path = _expert_data_path / env_name / test_flag
    elif data_type == "random":
        task_data_path = _expert_datarandom_path / env_name / test_flag
    elif data_type == "mixed":
        task_data_path = _expert_datamixed_path / env_name / test_flag
    elif data_type == "mixed_20_80":
        task_data_path = _expert_datamixed_20_80_path / env_name / test_flag
    elif data_type == "mixed_40_80":
        task_data_path = _expert_datamixed_40_80_path / env_name / test_flag
    elif data_type == "opticalflow":
        task_data_path = _expert_dataopticalflow_path / env_name / test_flag
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    return sorted(task_data_path.iterdir(), key=lambda x: int(x.stem))[:MAX_DATA_CHUNKS]


def get_experiment_dir(exp_name, current_save, step, loss_dir):
    if current_save:
        d = _current_experiment_results_path / f"{exp_name}_step_{step}_loss_{loss_dir}"
    else:
        d = _experiment_results_path / exp_name
    d.mkdir(exist_ok=True, parents=True)
    return d


def get_models_path(exp_name: str, current_save: bool = False, step: int = None, loss_dir: str = None):
    return get_experiment_dir(exp_name, current_save, step, loss_dir) / "idm_fdm.pt"


def get_latent_policy_path(exp_name, current_save: bool = False, step: int = None, loss_dir: str = None):
    return get_experiment_dir(exp_name, current_save, step, loss_dir) / "latent_policy.pt"


def get_decoded_policy_path(exp_name, current_save: bool = False, step: int = None, loss_dir: str = None):
    return get_experiment_dir(exp_name, current_save, step, loss_dir) / "decoded_policy.pt"