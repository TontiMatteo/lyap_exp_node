import numpy as np

def make_concentric_circles(n_samples, inner_range=(0.0, 0.75), outer_range=(1.25, 1.75), seed=42):
    np.random.seed(seed)
    
    # Split samples between inner and outer (adjusting for area if desired, 
    # but the paper usually uses a 1:2 ratio or equal split)
    n_inner = n_samples // 3
    n_outer = n_samples - n_inner
    
    # 1. Generate Inner Circle (Radius 0.0 to 0.75)
    r_inner = np.random.uniform(inner_range[0], inner_range[1], n_inner)
    theta_inner = np.random.uniform(0, 2*np.pi, n_inner)
    X_inner = np.column_stack([r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)])
    y_inner = -np.ones(n_inner)  # Label -1
    
    # 2. Generate Outer Annulus (Radius 1.0 to 1.5)
    r_outer = np.random.uniform(outer_range[0], outer_range[1], n_outer)
    theta_outer = np.random.uniform(0, 2*np.pi, n_outer)
    X_outer = np.column_stack([r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)])
    y_outer = np.ones(n_outer)   # Label 1
    
    # Combine and shuffle
    X = np.vstack([X_inner, X_outer])
    y = np.hstack([y_inner, y_outer])
    
    indices = np.random.permutation(n_samples)
    return X[indices], y[indices]

# Parameters to match the "Second Paper" setup:
# n_samples=3000 to match the 1000/2000 split in the notebook
X, y = make_concentric_circles(n_samples=1500, inner_range=(0.0, 0.75), outer_range=(1.25, 2))

# Save for your first model
np.savez(
    "circle_anode_style_val.npz",
    X=X,
    y=y
)