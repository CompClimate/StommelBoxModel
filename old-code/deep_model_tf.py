import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from keras import layers
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers as tfpl


def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.models.Sequential(
        [
            tfpl.DistributionLambda(
                lambda t: tfd.MultivariateNormalDiag(loc=tf.zeros(n), scale_diag=tf.ones(n))
            ),
        ]
    )
    return prior_model


def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.models.Sequential(
        [
            tfpl.VariableLayer(tfpl.MultivariateNormalTriL.params_size(n), dtype=dtype),
            tfpl.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model


def bayes_mlp(hidden_dim, num_layers):
    net = tfpl.DenseVariational(10)
    return net


X_train = np.linspace(-1, 1, 1000)[:, np.newaxis]
y_train = np.power(X_train, 3) + 0.1 * (2 + X_train) * np.random.randn(1000)[:, np.newaxis]

model = keras.models.Sequential(
    [
        tfpl.DenseVariational(
            units=8,
            input_shape=(1,),
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1 / X_train.shape[0],
            activation="sigmoid",
        ),
        tfpl.DenseVariational(
            units=tfpl.IndependentNormal.params_size(1),
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1 / X_train.shape[0],
        ),
        tfpl.IndependentNormal(1),
    ]
)


def nll(y_true, y_pred):
    return -y_pred.log_prob(y_true)


model.compile(loss=nll, optimizer=keras.optimizers.RMSprop(learning_rate=0.005))

# model.compile(
#     loss=keras.losses.MeanSquaredError(),
#     optimizer=keras.optimizers.RMSprop(learning_rate=0.005),
# )
model.fit(X_train, y_train, batch_size=16, epochs=100)

y_model = model(X_train)
y_hat = y_model.mean()
y_hat_m2std = y_hat - 2 * y_model.stddev()
y_hat_p2std = y_hat + 2 * y_model.stddev()

plt.scatter(X_train, y_train, alpha=0.2, label="data")
plt.plot(X_train, y_hat, color="red", alpha=0.8, label=r"model $\mu$")
plt.plot(X_train, y_hat_m2std, color="green", alpha=0.8, label=r"model $\mu \pm 2 \sigma$")
plt.plot(X_train, y_hat_p2std, color="green", alpha=0.8)
plt.show()
