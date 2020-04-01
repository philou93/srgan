import matplotlib.pyplot as plt
import numpy as np


def show_single_model_loss(losses, model_name, mean=True):
    losses = np.array(losses, dtype=np.float)
    if mean is True:
        losses = np.mean(losses, axis=1)
    else:
        losses = losses.flatten()
    plt.plot(losses)
    plt.ylabel('mse')
    plt.xlabel('step')
    plt.xticks(list(range(losses.shape[0])))
    plt.title(f'{model_name} losses')
    plt.show()
