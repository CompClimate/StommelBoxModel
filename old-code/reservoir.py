import matplotlib.pyplot as plt
import numpy as np
import reservoirpy as rpy
from reservoirpy.nodes import ESN, RLS, Reservoir, Ridge

rpy.verbosity(0)
rpy.set_seed(42)


def plot_states(states):
    plt.figure(figsize=(10, 3))
    plt.plot(states[:, :20])


t = np.linspace(0, 3 * np.pi, 200)
X = np.sin(t).reshape(-1, 1)
train_len = 100
X_train = X[:train_len]
y_train = X[1 : (train_len + 1)]

reservoir = Reservoir(100, lr=0.5, sr=0.9)
readout = Ridge(
    ridge=1e-1,
)

esn_model = reservoir >> readout
esn_model = esn_model.fit(X_train, y_train, warmup=10)

pred = []
x = X
for _ in range(10):
    Y_pred = esn_model.run(x)
    x = Y_pred
    pred += Y_pred.tolist()

t_cont = np.linspace(3 * np.pi, 10 * 3 * np.pi, 900)
X_cont = np.sin(t_cont)

plt.figure(figsize=(10, 3))

plt.plot(pred, label="Predicted", color="blue")

X_ls = X.reshape(-1, 1).tolist() + X_cont.reshape(-1, 1).tolist()
plt.plot(X_ls, label="Real", color="red")

plt.xlabel("$t$")
plt.legend()
plt.show()
