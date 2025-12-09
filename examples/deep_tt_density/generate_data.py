import torch
import numpy as np
import sys

def euler_maruyama(drift_func, diffusion_func, x0, t_start, t_end, dt):
    """
    Euler-Maruyama solver for SDEs.
    dX_t = f(X_t, t)dt + g(X_t, t)dW_t

    Args:
        drift_func: callable f(x, t) -> Tensor [batch, dim]
        diffusion_func: callable g(x, t) -> Tensor [batch, dim] (diagonal noise) or [batch, dim, dim]
        x0: Initial state [batch, dim]
        t_start: Start time
        t_end: End time
        dt: Time step

    Returns:
        trajectories: Tensor [steps, batch, dim]
        times: Tensor [steps]
    """
    device = x0.device
    
    # Ensure t_start and t_end are floats
    t_start = float(t_start)
    t_end = float(t_end)
    dt = float(dt)
    
    num_steps = int(round((t_end - t_start) / dt))
    times = torch.linspace(t_start, t_end, num_steps + 1, device=device)
    
    xs = [x0]
    x_curr = x0
    
    sqrtdt = np.sqrt(dt)
    
    for i in range(num_steps):
        t = times[i]
        
        # Compute drift and diffusion
        f_val = drift_func(x_curr, t)
        g_val = diffusion_func(x_curr, t)
        
        # Generate Wiener increment
        # Assuming diagonal noise if g_val matches x_curr shape
        # If g_val is scalar, it broadcasts
        dw = torch.randn_like(x_curr) * sqrtdt
        
        # Update step
        x_next = x_curr + f_val * dt + g_val * dw
        
        xs.append(x_next)
        x_curr = x_next
        
    return torch.stack(xs), times


class DoubleWellProcess:
    """
    Multi-dimensional Double Well process.
    Potential V(x) = sum_i alpha_i * (x_i^2 - 1)^2
    Drift f(x) = -grad V(x)
    """
    def __init__(self, dim, alpha=1.0, sigma=0.5):
        self.dim = dim
        # handle scalar or list alpha
        if np.isscalar(alpha):
            self.alpha = torch.tensor([alpha] * dim)
        else:
            self.alpha = torch.tensor(alpha)
            assert len(self.alpha) == dim
        
        self.sigma = sigma

    def drift(self, x, t):
        # x: [batch, dim]
        # grad V_i = 4 * alpha_i * x_i * (x_i^2 - 1)
        # drift = - grad V
        
        # Ensure alpha is on same device
        if self.alpha.device != x.device:
            self.alpha = self.alpha.to(x.device)
            
        return -4.0 * self.alpha * x * (x**2 - 1.0)

    def diffusion(self, x, t):
        return self.sigma


class OrnsteinUhlenbeckProcess:
    """
    Multi-dimensional Ornstein-Uhlenbeck process.
    dX = -theta * (x - mu) * dt + sigma * dW
    """
    def __init__(self, dim, theta=1.0, mu=0.0, sigma=0.5):
        self.dim = dim
        
        if np.isscalar(theta):
            self.theta = torch.tensor([theta] * dim)
        else:
            self.theta = torch.tensor(theta)
            
        if np.isscalar(mu):
            self.mu = torch.tensor([mu] * dim)
        else:
            self.mu = torch.tensor(mu)
            
        self.sigma = sigma

    def drift(self, x, t):
        # Ensure params are on same device
        if self.theta.device != x.device:
            self.theta = self.theta.to(x.device)
        if self.mu.device != x.device:
            self.mu = self.mu.to(x.device)
            
        return -self.theta * (x - self.mu)

    def diffusion(self, x, t):
        return self.sigma


def generate_sample(process_name, hyperparams, x_start, t_start, t_end, dt, sample_size):
    """
    Generates samples from a specified stochastic process.
    
    Args:
        process_name: 'double_well' or 'ornstein_uhlenbeck'
        hyperparams: dict containing process parameters (dim, alpha/theta, etc.)
        x_start: Optional starting point or trajectory.
                 - None: Samples x0 from standard normal or hyperparams['initial_dist']
                 - Tensor [dim]: Starts from this point, branches into sample_size trajectories
                 - Tensor [batch, dim]: Continues these trajectories (batch must match sample_size)
                 - Tensor [time, dim]: Continues from last point, branches into sample_size
                 - Tensor [time, batch, dim]: Continues from last point
        t_start: Start time of simulation
        t_end: End time of simulation
        dt: Time step
        sample_size: Number of trajectories to generate (if not determined by x_start input)
        
    Returns:
        trajectories: Tensor [steps, sample_size, dim]
    """
    dim = hyperparams.get('dim', 1)
    
    # Instantiate process
    if process_name == 'double_well':
        process = DoubleWellProcess(
            dim=dim,
            alpha=hyperparams.get('alpha', 1.0),
            sigma=hyperparams.get('sigma', 0.5)
        )
    elif process_name == 'ornstein_uhlenbeck':
        process = OrnsteinUhlenbeckProcess(
            dim=dim,
            theta=hyperparams.get('theta', 1.0),
            mu=hyperparams.get('mu', 0.0),
            sigma=hyperparams.get('sigma', 0.5)
        )
    else:
        raise ValueError(f"Unknown process name: {process_name}")

    # Determine x0
    if x_start is None:
        # Sample initial condition
        # Default to standard normal if not specified
        x0 = torch.randn(sample_size, dim)
        # If user wants specific initial distribution, could add logic here
    else:
        # Handle various shapes of x_start
        if not isinstance(x_start, torch.Tensor):
            x_start = torch.tensor(x_start, dtype=torch.float32)
            
        if x_start.ndim == 1:
            # Single point [dim] -> replicate
            x0 = x_start.unsqueeze(0).expand(sample_size, dim)
        elif x_start.ndim == 2:
            # Batch of points [batch, dim]
            if x_start.shape[0] == 1:
                x0 = x_start.expand(sample_size, dim)
            else:
                assert x_start.shape[0] == sample_size, f"x_start batch size {x_start.shape[0]} does not match sample_size {sample_size}"
                x0 = x_start
        elif x_start.ndim == 3:
            # Trajectory [time, batch, dim] -> take last point
            x_last = x_start[-1]
            if x_last.shape[0] == 1:
                 x0 = x_last.expand(sample_size, dim)
            else:
                assert x_last.shape[0] == sample_size, f"x_start batch size {x_last.shape[0]} does not match sample_size {sample_size}"
                x0 = x_last
        else:
             raise ValueError(f"Unsupported x_start shape: {x_start.shape}")

    # Run simulation
    trajectories, times = euler_maruyama(process.drift, process.diffusion, x0, t_start, t_end, dt)
    
    # If x_start was a trajectory, we might want to concatenate? 
    # The prompt says "continue with a sample", usually implies returning the full path or just the new part.
    # Returning the new part (including t_start which is x0) is standard for solvers.
    # The user can concatenate if needed.
    
    return trajectories

if __name__ == '__main__':
    # Example usage
    import matplotlib.pyplot as plt
    
    # Settings
    dim = 2
    sample_size_plot = 100     # For the trajectory fan-out visualization
    sample_size_hist = 10000   # For the histogram (independent sample)
    t1 = 0.0
    t2 = 2.0
    t3 = 4.0
    dt = 0.01
    
    processes = [
        ('double_well', {'dim': dim, 'alpha': 1.0, 'sigma': 0.5}, "Double Well"),
        ('ornstein_uhlenbeck', {'dim': dim, 'theta': 1.0, 'mu': 0.0, 'sigma': 0.5}, "Ornstein-Uhlenbeck")
    ]

    for proc_name, params, display_name in processes:
        print(f"\n--- Running {display_name} ({dim}D) ---")
        
        # --- Simulation 1: Trajectory Visualization (Fan-out) ---
        # 1. Generate a single partial trajectory (initial segment)
        print(f"Generating initial partial trajectory from t={t1} to t={t2}...")
        x_init = torch.zeros(dim) if proc_name == 'ornstein_uhlenbeck' else torch.ones(dim) * 0.5
        
        traj_part1 = generate_sample(proc_name, params, x_init, t1, t2, dt, 1)
        # traj_part1 shape: [steps1, 1, dim]
        
        print(f"Partial trajectory shape: {traj_part1.shape}")

        # 2. Continue with multiple trajectories for visualization
        print(f"Continuing with {sample_size_plot} samples from t={t2} to t={t3}...")
        
        traj_part2 = generate_sample(proc_name, params, traj_part1, t2, t3, dt, sample_size_plot)
        # traj_part2 shape: [steps2, sample_size_plot, dim]
        
        print(f"Continuation shape: {traj_part2.shape}")
        
        # --- Simulation 2: Histogram Data (Independent Sample) ---
        # 3. Generate a large independent sample from t=t1 to t=t3 starting from N(0,I)
        print(f"Generating {sample_size_hist} independent samples from t={t1} to t={t3} for histogram...")
        
        # x_start=None triggers standard normal initialization inside generate_sample
        traj_hist = generate_sample(proc_name, params, None, t1, t3, dt, sample_size_hist)
        final_positions = traj_hist[-1, :, :] # [sample_size_hist, dim]
        
        print(f"Histogram sample final shape: {final_positions.shape}")
        
        # --- Plotting ---
        try:
            fig = plt.figure(figsize=(12, 10))
            
            # Subplot 1 (Top-Left): Trajectories in 2D Plane
            ax1 = fig.add_subplot(2, 2, 1)
            
            # Plot initial trajectory (Part 1)
            ax1.plot(traj_part1[:, 0, 0].numpy(), traj_part1[:, 0, 1].numpy(), 'k-', linewidth=2, label='Initial Path', zorder=10)
            
            # Plot continuation trajectories (Part 2)
            # Plot a subset if plot sample is still large, but we set it to 100 which is fine
            for i in range(sample_size_plot):
                ax1.plot(traj_part2[:, i, 0].numpy(), traj_part2[:, i, 1].numpy(), alpha=0.1, color='blue')
            
            ax1.set_title(f"{display_name}: Trajectory Fan-out (x0 vs x1)")
            ax1.set_xlabel("x0")
            ax1.set_ylabel("x1")
            ax1.legend()
            ax1.grid(True)
            
            # Setup time arrays for time-series plots
            times1 = np.linspace(t1, t2, traj_part1.shape[0])
            times2 = np.linspace(t2, t3, traj_part2.shape[0])

            # Subplot 2 (Top-Right): Trajectory of Dim 0 over time
            ax2 = fig.add_subplot(2, 2, 2)
            
            ax2.plot(times1, traj_part1[:, 0, 0].numpy(), 'k-', linewidth=2, label='Initial Path', zorder=10)
            for i in range(sample_size_plot):
                 ax2.plot(times2, traj_part2[:, i, 0].numpy(), alpha=0.1, color='blue')
                 
            ax2.set_title("Dimension 0 vs Time")
            ax2.set_xlabel("Time")
            ax2.set_ylabel("x0")
            ax2.grid(True)

            # Subplot 3 (Bottom-Left): Trajectory of Dim 1 over time
            ax3 = fig.add_subplot(2, 2, 3)
            
            ax3.plot(times1, traj_part1[:, 0, 1].numpy(), 'k-', linewidth=2, label='Initial Path', zorder=10)
            for i in range(sample_size_plot):
                 ax3.plot(times2, traj_part2[:, i, 1].numpy(), alpha=0.1, color='blue')
                 
            ax3.set_title("Dimension 1 vs Time")
            ax3.set_xlabel("Time")
            ax3.set_ylabel("x1")
            ax3.grid(True)

            # Subplot 4 (Bottom-Right): 2D Histogram (Contourf) of Independent Sample
            ax4 = fig.add_subplot(2, 2, 4)
            
            counts, xedges, yedges = np.histogram2d(
                final_positions[:, 0].numpy(), 
                final_positions[:, 1].numpy(), 
                bins=50, density=True
            )
            xcenters = (xedges[:-1] + xedges[1:]) / 2
            ycenters = (yedges[:-1] + yedges[1:]) / 2
            X, Y = np.meshgrid(xcenters, ycenters)
            
            cs = ax4.contourf(X, Y, counts.T, levels=20, cmap='viridis')
            fig.colorbar(cs, ax=ax4)
            ax4.set_title(f"Independent Density at t={t3} (N={sample_size_hist})")
            ax4.set_xlabel("x0")
            ax4.set_ylabel("x1")
            
            plt.tight_layout()
            filename = f"{proc_name}_2d_demo.png"
            plt.savefig(filename)
            print(f"Saved plot to {filename}")
            plt.close()
            
        except Exception as e:
            print(f"Plotting failed: {e}")

