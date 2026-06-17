import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import DataLoader
from .lyap import NN_LyapExp

def avg_jacobian_norm(model, X):
    X = X.float().requires_grad_(True)

    z0 = torch.tanh(model.input_layer(X))
    f = model.odefunc(0, z0)               # Vector field evaluation
    jac_list = []

    for i in range(f.shape[0]):
        grad_f = torch.autograd.grad(f[i].sum(), z0, retain_graph=True, create_graph=False)[0]
        jac_list.append(grad_f[i].norm().item())

    return sum(jac_list)/len(jac_list)


def margin_accuracy(model, dataloader, margin=0.3):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.float()
            y = y.float()

            pred = model(x).squeeze()
            confident = (y * pred > margin)
            # confident = (torch.sign(pred) == torch.sign(y.squeeze()))

            correct += confident.sum().item()
            total += y.numel()

    return correct/total


def train(model: nn.Module,
          dataloader: DataLoader,
          val_dataloader: DataLoader,
          loss_fn: nn.Module,
          acc_target: float = 0.95,
          eval_every: int = 20,
          seed: int = 42,
          epochs: int = 2500,):
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    hits = 0
    required_hits = 3

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.float()
            y_batch = y_batch.float()

            # Forward pass
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch.squeeze())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x_batch.size(0)

        # Average over all of the dataset
        epoch_loss /= len(dataloader.dataset)

        if epoch % eval_every == 0:
            val_acc = margin_accuracy(model, val_dataloader, margin=0.5)
            print(f"Epoch {epoch} | Train Loss {epoch_loss:.6f} | Validation Accuracy {val_acc:.6f}")

            if val_acc > acc_target:
                hits += 1
                if hits >= required_hits:
                    print(f"Early stopping: reached {acc_target*100:.1f}% validation accuracy")
                    break
            else:
                hits = 0


def jacobian_node(model: nn.Module,
                  x: torch.tensor):
    
    x = x.clone().detach().requires_grad_(True)
    y = model(x)  # shape [1, output_dim]

    if y.ndim == 0:
        y = y.view(1, 1) 
    elif y.ndim == 1:          # single output
        y = y.unsqueeze(0)    # make it [1, 1]

    input_dim = x.shape[1]
    output_dim = y.shape[1]
    J = torch.zeros(output_dim, input_dim, device=x.device, dtype=x.dtype)

    # For each output component, compute gradient w.r.t input
    for i in range(output_dim):
        grad = torch.autograd.grad(y[0, i], x, retain_graph=True, allow_unused=True)[0]
        if grad is None:
            grad = torch.zeros_like(x)
        J[i, :] = grad.reshape(-1)
    return J

def lyapunov_autograd(x_vals: np.array,
                         y_vals: np.array,
                         model: nn.Module,
                         mean: float,
                         std: float,
                         t0: float = 0.0,
                         t1: float = 1.0
):
    lyap_grid = np.zeros((len(x_vals), len(y_vals)))
    lyap_grid_min = np.zeros((len(x_vals), len(y_vals)))

    nn_lyap = NN_LyapExp(model)
    
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):

            # Create 1x2 input
            x0 = torch.tensor([[x, y]], dtype=torch.float32)

            # Standardize using mean and std of the dataset
            x0 = ((x0 - mean) / std).float()

            # Compute Jacobian
            J = nn_lyap.jacobian_flow(x0, t0=t0, t1=t1)

            # Singular values
            svals = torch.linalg.svdvals(J)

            # Largest Lyapunov proxy
            lyap_grid[j, i] = torch.log(svals.max()).item()
            lyap_grid_min[j, i] = torch.log(svals.min()).item()

    return lyap_grid, lyap_grid_min

def lyapunov_autograd_full(
    x_vals,
    y_vals,
    model,
    mean,
    std,
    t0=0.0,
    t1=1.0
):

    ftle_1 = np.zeros((len(y_vals), len(x_vals)))
    ftle_2 = np.zeros((len(y_vals), len(x_vals)))
    ftle_3 = np.zeros((len(y_vals), len(x_vals)))

    nn_lyap = NN_LyapExp(model)

    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):

            # Original 2D point
            xy = torch.tensor([[x, y]], dtype=torch.float32)

            # Standardize original coordinates
            xy = ((xy - mean) / std).float()

            # Initial augmented coordinate = 0
            aug = torch.zeros((1, 1))

            # Full 3D initial condition
            z0 = torch.cat([xy, aug], dim=1)

            # Full 3x3 Jacobian
            J = nn_lyap.jacobian_flow(z0, t0=t0, t1=t1)

            # Singular values
            svals = torch.linalg.svdvals(J)

            # Sort descending
            svals, _ = torch.sort(svals, descending=True)

            ftle_1[j, i] = torch.log(svals[0]).item()
            ftle_2[j, i] = torch.log(svals[1]).item()
            ftle_3[j, i] = torch.log(svals[2]).item()

    return ftle_1, ftle_2, ftle_3
