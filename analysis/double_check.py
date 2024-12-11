import numpy as np

data = np.load("datasets/TimeDrift/241209.npz")
X = data["X"]
y = data["y"]

new_y = np.random.randint(0, 105, size=y.shape)


print(X)
print(y)
print(new_y)
assert y.min() == new_y.min()
assert y.max() == new_y.max()
assert len(np.unique(y)) == len(np.unique(new_y))
print("shape:", X.shape, y.shape)

np.savez("datasets/TimeDrift/debug.npz", X=X, y=new_y)