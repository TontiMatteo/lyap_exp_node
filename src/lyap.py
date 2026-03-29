import torch
import torch.nn as nn
import numpy as np

import inspect

class LyapunovExponents():
    """
    This class computes Lyapunov exponents ifor discrete maps or continuous time systems.
    The algorithm used is the Benettin Algorithm.
    Use the continuous flag to switch between modes.
    Supports euler or rk4 integration for continuous systems
    """

    def __init__(self, dim, n_vectors=None, dt=1.0, continuous=False, integrator='euler'):
        """
        Input:
            dim: dimensionality of the system
            n_vectors: number of perturbation vectors, default to dim
            dt: time-step, default to 1
            Continuous: true for continuous time ODE, false for discrete maps
            integrator: 'euler' or 'rk4', only used for continuous systems.
        """
        self.dim = dim
        self.n_vectors = n_vectors if n_vectors is not None else dim
        self.dt = dt
        self.continuous = continuous
        self.integrator = integrator.lower()

    def compute(self, func, x0, jac=None, T=100.0, orthonormalize=True, return_ftle=False, sample_every=100):
        """
        Compute Lyapunov exponents.

        func: vector field or map
        x0: initial condition (shape = dim)
        T: total integration time (continuous) or number of steps (discrete)
        orthonormalize: whether to use QR decomposition
        return_ftle: if True, return FTLE over time
        sample_every: steps interval to record FTLE for plotting
        """
        x = x0
        n = self.dim
        m = self.n_vectors
        Q = torch.eye(n, m, dtype=x.dtype, device=x.device)
        log_norms = torch.zeros(m, dtype=x.dtype, device=x.device)
        
        # Steps for integration
        steps = int(T / self.dt) if self.continuous else T

        func = self._wrap_func(func)

        # For FTLE recording
        ftle_values = []
        time_values = []

        for step in range(steps):
            t = step * self.dt
            f_eval = func(x, t)

            if self.continuous:
                if self.integrator == 'rk4':
                    Q, x = self._rk4_step(func, x, Q, t, jac)
                elif self.integrator == 'euler':
                    J = jac(x, t) if jac is not None else self.jacobian(func, x, t)
                    
                    Q = Q + self.dt * (J @ Q)
                    x = x + self.dt * f_eval
                else:
                    raise ValueError("Integrator must be 'euler' or 'rk4'")
            else:
                # discrete map
                J = jac(x, t) if jac is not None else self.jacobian(func, x, t)
                Q = J @ Q
                x = func(x, t)

            # QR orthonormalization
            if orthonormalize:
                if self.n_vectors > 1:
                    Q, R = torch.linalg.qr(Q)
                    log_norms += torch.log(torch.abs(torch.diag(R)))
                else:
                    # 1D: Q is scalar, no QR needed
                    log_norms += torch.log(torch.abs(Q.flatten()))
                    Q[:] = 1.0  # reset for numerical stability
            else:
                norms = torch.linalg.norm(Q, dim=0)
                log_norms += torch.log(norms)
                Q = Q / norms

            # Record FTLE
            if return_ftle and (step + 1) % sample_every == 0:
                t = (step + 1) * self.dt if self.continuous else step + 1
                ftle_values.append(log_norms.clone() / t)
                time_values.append(t)

        # Final exponent
        if self.continuous:
            lyap_exp = log_norms / (steps * self.dt)
        else:
            lyap_exp = log_norms / T

        if return_ftle:
            return lyap_exp, torch.tensor(time_values), torch.stack(ftle_values)
        else:
            return lyap_exp
            

    def _rk4_step(self, func, x, Q, t, jac=None):
        """
        One RK4 integration step for both x and Q.
        """
        dt = self.dt

        f1 = func(x, t)
        J1 = jac(x, t) if jac is not None else self.jacobian(func, x, t)
        k1Q = J1 @ Q

        x2 = x + 0.5 * dt * f1
        f2 = func(x2, t + 0.5*dt)
        J2 = jac(x2, t + 0.5*dt) if  jac is not None else self.jacobian(func, x2, t + 0.5*dt)
        k2Q = J2 @ (Q + 0.5 * dt * k1Q)

        x3 = x + 0.5 * dt * f2
        f3 = func(x3, t + 0.5*dt)
        J3 = jac(x3, t + 0.5*dt) if jac is not None else self.jacobian(func, x3, t + 0.5*dt)
        k3Q = J3 @ (Q + 0.5 * dt * k2Q)

        x4 = x + dt * f3
        f4 = func(x4, t + dt)
        J4 = jac(x4, t + dt) if jac is not None else self.jacobian(func, x4, t + dt)
        k4Q = J4 @ (Q + dt * k3Q)

        # RK4 updates
        Q_next = Q + (dt / 6.0) * (k1Q + 2 * k2Q + 2 * k3Q + k4Q)
        x_next = x + (dt / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
        return Q_next, x_next

    @staticmethod
    def jacobian(func, x, t):
        """
        Compute the Jacobian of the function through finite differences.
        Inputs:

            x: state vector wrt which we are computing the Jacobian
            eps: step size

        Output:

            J: Jacobian matrix of the function
        """
        x = x.clone().detach().requires_grad_(True)
        y = func(x, t)

        n = y.numel()
        m = x.numel()

        J = torch.zeros(n, m, device=x.device, dtype=x.dtype)
        for i in range(n):
            grad_i = torch.autograd.grad(y.flatten()[i], x, retain_graph=True)[0]
            J[i, :] = grad_i.reshape(-1)
        
        return J
    
    def jacobian_fd(func, x, eps=1e-6):
        """
        Finite-difference Jacobian of func(x)

        Inputs
            func : function R^m -> R^n
            x    : tensor (m,)
            eps  : finite difference step

        Output
            J : tensor (n, m)
        """

        x = x.detach()
        m = x.numel()

        y0 = func(x)
        n = y0.numel()

        J = torch.zeros(n, m, device=x.device, dtype=x.dtype)

        for i in range(m):
            dx = torch.zeros_like(x)
            dx[i] = eps

            y_plus = func(x + dx)
            y_minus = func(x - dx)

            J[:, i] = ((y_plus - y_minus) / (2 * eps)).reshape(-1)

        return J
    
    def _wrap_func(self, func):
        """
        Returns a function that always accepts (x, t) as a input
        """
        sig = inspect.signature(func)

        if len(sig.parameters) == 1:
            def wrapped(x, t):
                return func(x)
        else:
            wrapped = func

        return wrapped
    

class NN_LyapExp:
    """
    Compute Lyapunov exponents of a neural network.
    Supports fully-connected networks with any input dimension.
    """

    def __init__(self, model):
        self.model = model

    def jacobian(self, x):
        """
        Compute the Jacobian of model(x) with respect to x.
        x: torch.tensor with shape [1, input_dim]
        returns: Jacobian matrix of shape [input_dim, input_dim]
        """
        x = x.clone().detach().requires_grad_(True)
        y = self.model(x)  # shape [1, output_dim]

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
    
    def jacobian_flow(self, x, t0=0.0, t1=1.0):
        x = x.clone().detach().requires_grad_(True)

        def phi(z):
            return self.flow_map(z, t0=t0, t1=t1)
        
        J = torch.autograd.functional.jacobian(phi, x)
        return J.squeeze()
    
    def jacobian_flow_fd(self, x, t0=0.0, t1=1.0, eps=1e-6):
        """
        Finite-difference Jacobian of the flow map Φ_T(x)
        """

        x = x.detach()
        m = x.numel()

        y0 = self.flow_map(x, t0=t0, t1=t1)
        n = y0.numel()

        J = torch.zeros(n, m, device=x.device, dtype=x.dtype)

        for i in range(m):
            dx = torch.zeros_like(x)
            dx[i] = eps

            y_plus = self.flow_map(x + dx, t0=t0, t1=t1)
            y_minus = self.flow_map(x - dx, t0=t0, t1=t1)

            J[:, i] = ((y_plus - y_minus) / (2 * eps)).reshape(-1)

        return J

    def lyapunov_spectrum(self, x0, T=1000, n_lyap=None):
        """
        Compute the full Lyapunov spectrum of the NN starting from x0.
        x0: torch.tensor [1, input_dim]
        T: number of iterations
        n_lyap: number of exponents to compute
        """
        n = x0.numel()
        if n_lyap is None:
            n_lyap = n

        Q = torch.eye(n)[:, :n_lyap]
        log_diag_R = torch.zeros(n_lyap)

        x = x0.clone()
        for t in range(T):
            # Compute Jacobian at current state
            J = self.jacobian(x)

            # Forward pass
            x = self.model(x)

            # Tangent propagation
            Q = J @ Q

            # QR decomposition to re-orthonormalize
            Q, R = torch.linalg.qr(Q)
            log_diag_R += torch.log(torch.abs(torch.diag(R)) + 1e-12)  # small epsilon for stability

        return (log_diag_R / T).detach().numpy()
    
    def max_lyapunov(self, x0, T=1000):
        """
        Compute only the maximum FTLE. It works also for non-square jacobians
        """
        x = x0.clone()
        v = torch.randn_like(x)  # random initial perturbation
        v = v / torch.norm(v)

        log_sum = 0.0
        for _ in range(T):
            J = self.jacobian(x)
            x = self.model(x)

            v = J @ v.T
            norm_v = torch.norm(v)
            v = v / norm_v
            log_sum += torch.log(norm_v + 1e-12)

        return (log_sum / T).item()
    
    def flow_map(self, x, t0=0.0, t1=1.0):
        """
        Flow map Φ^T(x)
        """
        return self.model(x, t0=t0, t1=t1)
    

