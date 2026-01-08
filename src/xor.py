import numpy as np

def make_xor(n_samples, noise=0.1, seed=42):
    np.random.seed(seed)

    X = np.random.rand(n_samples, 2)
    y = ((X[:, 0] > 0.5) ^ (X[:, 1] > 0.5)).astype(int)

    # Add Gaussian noise
    X += noise * np.random.randn(n_samples, 2)

    return X, y

n_samples = 500

X, y = make_xor(500)

np.savez(
    "xor_dataset_n0p1.npz",
    X=X,
    y=y
)