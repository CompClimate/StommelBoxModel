import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

import box_model
import deep_model

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
    _, _, F, DeltaS, DeltaT, q = box_model.get_time_series(
        model,
        time_max,
        forcing=cfg['data'].get('forcing', 'sinusoidal'),
        forcing_kwargs=cfg['data'].get('forcing_kwargs', dict()),
        fig=fig,
        ax=ax,
    )

    feats = {
        'F': F,
        'DeltaS': DeltaS,
        'DeltaT': DeltaT,
    }

    if cfg['model']['return_plot']:
        plt.show()

    X = np.hstack(
        [feats[name].reshape(-1, 1) for name in cfg['data']['input_features']]
    )
    y = q

    X, y = X.astype(np.float32), y.astype(np.float32)

    X_train, X_test, y_train_, y_test_ = \
        train_test_split(
            X, y, test_size=cfg['data']['test_size'], shuffle=False,
        )

    cfg['model']['input_dim'] = len(cfg['data']['input_features'])
    deep_model.train(
        X_train, y_train_,
        X_test, y_test_,
        cfg['model'],
    )

    if cfg['model']['return_plot']:
        plt.show()


if __name__ == '__main__':
    main()
