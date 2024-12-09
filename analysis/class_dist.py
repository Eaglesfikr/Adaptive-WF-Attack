import numpy as np
import os

for file_name in ["0313.npz", "0327.npz", "0410.npz", "0709.npz", "1201.npz"]:
    data = np.load(os.path.join("datasets/TimeDrift", file_name))
    y = data["y"]
    print("loading...", file_name)
    print(y.shape)

    for i in range(int(y.max())+1):
        if (y==i).sum() < 60:
            print(f"{i}: {(y==i).sum()}")


