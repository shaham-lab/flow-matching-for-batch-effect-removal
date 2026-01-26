"""ODE integration for flow matching inference."""

import torch
from typing import Literal


def euler_step(
    z: torch.Tensor,
    t: float,
    dt: float,
    vector_field: callable
) -> torch.Tensor:
    """
    Single Euler step for ODE integration.
    
    dz/dt = v_theta(z, t)
    z_{t+dt} = z_t + dt * v_theta(z_t, t)
    
    Args:
        z: Current latent point [batch_size, latent_dim]
        t: Current time
        dt: Time step
        vector_field: Function v_theta(z, t) -> [batch_size, latent_dim]
        
    Returns:
        Updated latent point
    """
    v = vector_field(z, torch.full((z.shape[0],), t, device=z.device))
    return z + dt * v


def rk4_step(
    z: torch.Tensor,
    t: float,
    dt: float,
    vector_field: callable
) -> torch.Tensor:
    """
    Single RK4 step for ODE integration.
    
    Args:
        z: Current latent point [batch_size, latent_dim]
        t: Current time
        dt: Time step
        vector_field: Function v_theta(z, t) -> [batch_size, latent_dim]
        
    Returns:
        Updated latent point
    """
    k1 = vector_field(z, torch.full((z.shape[0],), t, device=z.device))
    k2 = vector_field(z + dt * k1 / 2, torch.full((z.shape[0],), t + dt / 2, device=z.device))
    k3 = vector_field(z + dt * k2 / 2, torch.full((z.shape[0],), t + dt / 2, device=z.device))
    k4 = vector_field(z + dt * k3, torch.full((z.shape[0],), t + dt, device=z.device))
    
    return z + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_flow(
    z0: torch.Tensor,
    vector_field: callable,
    n_steps: int = 100,
    method: Literal["euler", "rk4"] = "euler"
) -> torch.Tensor:
    """
    Integrate learned vector field from t=0 to t=1.
    
    z_corr = z0 + ∫[0 to 1] v_theta(z_t, t) dt
    
    Args:
        z0: Initial latent points [batch_size, latent_dim]
        vector_field: Learned vector field function
        n_steps: Number of integration steps
        method: Integration method ('euler' or 'rk4')
        
    Returns:
        Corrected latent points [batch_size, latent_dim]
    """
    z = z0.clone()
    dt = 1.0 / n_steps
    
    step_fn = euler_step if method == "euler" else rk4_step
    
    for i in range(n_steps):
        t = i * dt
        z = step_fn(z, t, dt, vector_field)
    
    return z

