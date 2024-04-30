# Atlantic Meridional Overturning Circulation (AMOC) Prediction using Machine Learning

This is the code corresponding to the paper ["The Importance of Architecture Choice in Deep Learning for Climate Applications"](https://arxiv.org/abs/2402.13979).
We use Python 3.10 along with [Hydra](https://hydra.cc/) to provide reproducible and configurable experiments.

Out of the box, the following types of neural networks are implemented:
- Bayesian Neural Network (BNN)
- Multi-Layer Perceptron (MLP) with ReLU activations
- Deep Ensemble (`k` independently trained MLPs whose outputs are averaged)
- Convolutional Neural Network (CNN)
- Recurrent Neural Network (RNN).
The model configurations are defined in `configs/model/` and the box model configuration is defined in `configs/box_model/`.

# Training

Training is invoked using a simple command-line call:
```python
python src/train.py experiment=... <kwargs>
```
By default, log files for an experiment are saved in `logs/`. Default parameters when `<kwargs>` is empty are found in `configs/train.yaml`.

# Evaluation

Evaluating a PyTorch model according to a set of metrics is similar to invoking training:
```python
python src/eval.py experiment=... <kwargs>
```

# Scenario Generation

Almost arbitrary forcing scenarios are possible to generate as long as they can be defined as an interpolation between a time step `time_max`
and the current time step `t`. Here we pre-define a selection of scenarios that are used in the paper. For pre-defined forcing scenarios, see `configs/s_forcing` and `configs/t_forcing`, respectively.

# Explainability

Explaining model output using algorithms from the `captum` package is available by defining an algorithm in `configs/explainability/` and setting `explainability=<algorithm>` in the training `<kwargs>`.
