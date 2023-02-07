import numpy as np

data = np.array([0.675, 0.675, 0.4])
m = data.mean()
h = data.std() * 2

print(f"{m:.02f} $\pm$ {h:.02f}")