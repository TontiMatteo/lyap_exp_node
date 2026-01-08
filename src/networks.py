import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint

class NeuralODE1D(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, t, x):
        return self.net(x)


class SimpleFNN(nn.Module):

    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, n),
            nn.Tanh(),
            nn.Linear(n,1),
            nn.Tanh()
        )

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=1.0 / layer.in_features**0.5)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)

    
class SimpleFNN_Truncated(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.net = base_model.net[:2]  # up to first Tanh

    def forward(self, x):
        return self.net(x)
    
    
def train_loop(dataloader, model, loss_fn, optimizer, epochs=100, device='cpu'):
    batch_size = len(dataloader.dataset)

    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch, (X,y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)


            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch % 100 == 0:
                current = batch * len(X)
                print(f"Epoch {epoch+1:03d} | Batch {batch:03d} | Loss: {loss.item():.6f}")
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1:03d} complete — Average loss: {avg_loss:.6f}\n")

    print("Training finished!")
            

class ODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.Tanh()
        )

    def forward(self, t, z):
        return self.net(z)
    

class NeuralODEClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.odefunc = ODEFunc(hidden_dim=hidden_dim)
        self.final_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        z0 = torch.tanh(self.input_layer(x))
        tspan = torch.tensor([0, 1], dtype=torch.float32, device=x.device)     # integrate from t=0 to t=1
        zT = odeint(self.odefunc, z0, tspan, method="dopri5")[-1]
        out = torch.tanh(self.final_layer(zT))
        return out.squeeze()
    
    def get_hidden_trajectory(self, x, t_eval):
        z0 = torch.tanh(self.input_layer(x))
        z_traj = odeint(self.odefunc, z0, t_eval)
        return z_traj

class NeuralODE_Truncated(nn.Module):
    """
    Returns the hidden state z(T) of the Neural ODE,
    excluding the final linear output layer.
    """
    def __init__(self, neural_ode_model, t0=0.0, t1=1.0):
        super().__init__()
        self.model = neural_ode_model
        self.tspan = torch.tensor([t0, t1])

    def forward(self, x):
        # x: shape [B, input_dim]
        z0 = torch.tanh(self.model.input_layer(x))           # initial hidden state
        zT = odeint(self.model.odefunc, z0, self.tspan)[-1]  # integrate to T
        return zT                                            # do NOT apply final layer
    

def init_weights(module, init_type="xavier_normal", gain=1.0):
    if isinstance(module, nn.Linear):
        if init_type == "gaussian":
            nn.init.normal_(module.weight, mean=0.0, std=gain)
        elif init_type == "uniform":
            nn.init.uniform_(module.weight, a=-gain, b=gain)
        elif init_type == "xavier_normal":
            nn.init.xavier_normal_(module.weight, gain=gain)
        elif init_type == "xavier_uniform":
            nn.init.xavier_uniform_(module.weight, gain=gain)
        elif init_type == "kaiming_normal":
            nn.init.kaiming_normal_(module.weight, nonlinearity="tanh")
        elif init_type == "orthogonal":
            nn.init.orthogonal_(module.weight, gain=gain)

        else:
            raise ValueError(f"Unknown init type: {init_type}")
        
        if module.bias is not None:
            nn.init.zeros_(module.bias)