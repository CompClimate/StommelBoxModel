from pathlib import Path

config_dir = Path("config")
model_types = ["mlp", "ensemble", "bnn"]
autoregressive_model_types = ["mlp", "ensemble", "bnn", "conv", "rnn", "gru", "lstm"]
large_forcings = ["linear", "sinusoidal"]
small_forcings = ["low_period", "stationary", "nonstationary", "nonstationary_onesided"]

for forcing_dir1 in large_forcings:
    path1 = config_dir / forcing_dir1

    for forcing_dir2 in small_forcings:
        path2 = path1 / forcing_dir2
        path2.mkdir(parents=True, exist_ok=True)

        # Create shared data config
        (path2 / "data.yaml").touch(exist_ok=True)

        # Create `autoregressive` subfolder
        autoreg_path = path2 / "autoregressive"
        autoreg_path.mkdir(exist_ok=True)

        # Create autoregressive subfolder model folders
        for model_type in autoregressive_model_types:
            autoreg_path_model_type = autoreg_path / model_type
            autoreg_path_model_type.mkdir(exist_ok=True)

            # Create model file
            (autoreg_path_model_type / "model.yaml").touch(exist_ok=True)

            # Create main config file
            (autoreg_path_model_type / f"{model_type}.yaml").touch(exist_ok=True)

        for model_type in model_types:
            path3 = path2 / model_type
            path3.mkdir(exist_ok=True)

            # Create model file
            (path3 / "model.yaml").touch(exist_ok=True)

            # Create main config file
            (path3 / f"{model_type}.yaml").touch(exist_ok=True)
