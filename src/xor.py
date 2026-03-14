import numpy as np

def make_xor(n_samples, noise=0.1, seed=42):
    np.random.seed(seed)

    X = np.random.rand(n_samples, 2)
    y = ((X[:, 0] > 0.5) ^ (X[:, 1] > 0.5)).astype(int)
    y = 2*y - 1  # convert 0/1 to -1/1

    # Add Gaussian noise
    X += noise * np.random.randn(n_samples, 2)

    return X, y

n_samples = 250

X, y = make_xor(250, noise=0.01)

np.savez(
    "xor_dataset_n0p1_val.npz",
    X=X,
    y=y
)