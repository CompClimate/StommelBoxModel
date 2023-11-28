import matplotlib.pyplot as plt
import numpy as np
import rctorch.data as data
from matplotlib.pyplot import cm
from rctorch import RcBayesOpt, RcNetwork
from rctorch.data import final_figure_plot as phase_plot

plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.size": 22})
plotting_args = {"ylabel_pred": r"$x$, $p$"}

###

# def logistic(r, x):
#     return r * x * (1 - x)

# fig, ax = plt.subplots()

# n = 10000
# r = np.linspace(2.5, 4.0, n)
# iterations = 1000
# last = 100
# x = 1e-5 * np.ones(n)

# for i in range(iterations):
#     x = logistic(r, x)
#     if i >= (iterations - last):
#         ax.plot(r, x, ',k', alpha=.25)

# ax.set_xlabel(r'\(r\)')
# ax.set_ylabel(r'\(x\)')
# ax.set_title("Bifurcation diagram")
# fig.tight_layout()

# plt.show()
# exit()

###

fp_data = data.load("forced_pendulum", train_proportion=0.2, dt=np.pi / 20)
force_train, force_test = fp_data["force"]
target_train, target_test = fp_data["target"]

hps = {
    "n_nodes": 202,
    "log_connectivity": 0.4071449746896983,
    "spectral_radius": 1.1329107284545898,
    "regularization": 1.6862021450927922,
    "leaking_rate": 0.009808523580431938,
    "bias": 0.48509588837623596,
}

rcnet = RcNetwork(**hps, random_state=210, feedback=True)
rcnet.fit(y=target_train)
score, prediction = rcnet.test(y=target_test)

pred = rcnet.predict(10)
print(f"MSE: {score:.3f}")

rcnet.combined_plot(**plotting_args)

# colors = {
#     "color_rc": "brown",
#     #'color_gt' : "midnightblue",
#     "color_noise": "peru",
#     "linewidth": 1,
#     "alpha": 0.9,
#     "noisy_alpha": 0.4,
#     "noisy_s": 1,
#     "pred_linestyle": "-.",
#     "color_map": cm.afmhot_r,
# }

# phase_plot(
#     target_test,
#     None,
#     prediction,
#     **colors,
#     label_fontsize=25,
#     figsize=(9, 4.5),
#     # tick_fontsize = 22,
# )

bounds_dict = {
    "log_connectivity": (-2.5, -0.1),
    "spectral_radius": (0.1, 3),
    "n_nodes": (200, 202),
    "log_regularization": (-3, 1),
    "leaking_rate": (0, 0.2),
    "bias": (-1, 1),
}

rc_specs = {
    "feedback": True,
    "reservoir_weight_dist": "uniform",
    "output_activation": "tanh",
    "random_seed": 209,
}

rc_bo = RcBayesOpt(
    bounds=bounds_dict,
    scoring_method="nmse",
    n_jobs=4,
    cv_samples=1,
    initial_samples=25,
    **rc_specs,
)
opt_hps = rc_bo.optimize(
    n_trust_regions=4,
    max_evals=500,
    X=force_train,
    scoring_method="nmse",
    y=target_train,
)

my_rc2 = RcNetwork(**opt_hps, random_state=210, feedback=True)
my_rc2.fit(X=force_train, y=target_train)

score, prediction = my_rc2.test(X=force_test, y=target_test)
print(f"MSE {score:.3f}")
my_rc2.combined_plot(**plotting_args)

colors = {
    #'color_gt' : "midnightblue",
    "color_rc": "brown",
    "color_noise": "peru",
    "linewidth": 1,
    "alpha": 0.9,
    "noisy_alpha": 0.4,
    "noisy_s": 1,
    "pred_linestyle": "-.",
    "color_map": cm.afmhot_r,
}

phase_plot(
    test_gt=target_test,
    noisy_test_gt=None,
    rc_pred=prediction,
    **colors,
    label_fontsize=25,
    figsize=(9, 4.5),
)

plt.show()
