import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

import box_model
import deep_model
from deep_model import sliding_windows

plt.rcParams.update({
    'text.usetex': True,
	'text.latex.preamble': r"""
\usepackage{amsmath}
\usepackage{amssymb}
"""
})


@hydra.main(version_base=None)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    time_max = 150000 * box_model.YEAR
    model = box_model.BoxModel(
        **cfg['box_model'],
    )

    fig, ax = None, None
    _, _, F, X, y = box_model.get_time_series(
        model, time_max, forcing=cfg['data'].get('forcing', 'sinusoidal'), fig=fig, ax=ax,
    )

    X, y = X.astype(np.float32), y.astype(np.float32)
    X, y = sliding_windows(y, seq_length=cfg['data']['seq_len'])

    X_train, X_test, y_train_, y_test_ = \
        train_test_split(
            X, y, test_size=cfg['data']['test_size'], shuffle=False,
        )

    deep_model.pytorch_train(
        X_train, y_train_,
        X_test, y_test_,
        cfg['model'],
    )

    if cfg['model']['return_plot']:
        plt.show()


if __name__ == '__main__':
    main()
