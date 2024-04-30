import hydra


def get_working_dir():
    return hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
