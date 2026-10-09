import os
import time
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def main():
    start_wall = time.time()
    
    # 1. Load dataset from environment variable
    mat_path = os.environ["SILVERBOX_MAT"]
    mat_data = scipy.io.loadmat(mat_path)
    
    # Channels V1: u(t) (input), V2: y(t) (output)
    # The mat file structure depends on how scipy loads it
    v1 = mat_data['V1'].flatten()
    v2 = mat_data['V2'].flatten()
    
    # Excerpt extraction: indices 42650 to 46746 (length 4096)
    start_idx = 42650
    length = 4096
    u_excerpt = v1[start_idx : start_idx + length]
    y_excerpt = v2[start_idx : start_idx + length]
    
    # Center the signals (subtract mean of the excerpt)
    u_mean = np.mean(u_excerpt)
    y_mean = np.mean(y_excerpt)
    u_centered = u_excerpt - u_mean
    y_centered = y_excerpt - y_mean
    
    # Split train and test
    train_len = 3072
    test_len = 1024
    
    u_train = u_centered[:train_len]
    y_train = y_centered[:train_len]
    u_test = u_centered[train_len:]
    y_test = y_centered[train_len:]
    
    dt = 1.0 / 610.35
    
    # Convert to torch tensors
    device = torch.device("cpu") # CPU is robust and fast enough for 3000 steps
    u_train_t = torch.tensor(u_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    u_full_t = torch.tensor(u_centered, dtype=torch.float32, device=device)
    
    # 2. Define the Neural Network for acceleration alpha(y, v, u)
    class DuffingNet(nn.Module):
        def __init__(self, hidden_dim=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(3, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1)
            )
            
        def forward(self, y, v, u):
            # y, v, u can be batches or scalars
            inputs = torch.stack([y, v, u], dim=-1)
            return self.net(inputs).squeeze(-1)

    model = DuffingNet(hidden_dim=64).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # 3. RK4 integration step function in PyTorch
    def rk4_step(y, v, u_val, model, dt):
        # k1
        a1 = model(y, v, u_val)
        dy1 = v
        dv1 = a1
        
        # k2
        y2 = y + 0.5 * dt * dy1
        v2 = v + 0.5 * dt * dv1
        # if u needs interpolation, here u is sampled at discrete steps. For simplicity, use u_val at step.
        a2 = model(y2, v2, u_val)
        dy2 = v2
        dv2 = a2
        
        # k3
        y3 = y + 0.5 * dt * dy2
        v3 = v + 0.5 * dt * dv2
        a3 = model(y3, v3, u_val)
        dy3 = v3
        dv3 = a3
        
        # k4
        y4 = y + dt * dy3
        v4 = v + dt * dv3
        a4 = model(y4, v4, u_val)
        dy4 = v4
        dv4 = a4
        
        y_next = y + (dt / 6.0) * (dy1 + 2.0*dy2 + 2.0*dy3 + dy4)
        v_next = v + (dt / 6.0) * (dv1 + 2.0*dv2 + 2.0*dv3 + dv4)
        return y_next, v_next

    # 4. Training Strategy with rollout windows
    # Window length for training rollouts
    window_len = 64
    num_epochs = 800
    max_time = 140.0 # seconds budget for training
    
    # Initial velocity for training start: finite difference on training data
    v_train_init = (y_train[1] - y_train[0]) / dt
    
    print("Starting training...")
    model.train()
    for epoch in range(num_epochs):
        if time.time() - start_wall > max_time:
            print(f"Time budget reached at epoch {epoch}. Stopping training.")
            break
            
        optimizer.zero_grad()
        
        # Random starting point in training data ensuring a full window
        start_idx_roll = np.random.randint(0, train_len - window_len - 1)
        
        # Initialize rollout state from ground truth at start_idx_roll
        y_curr = y_train_t[start_idx_roll]
        if start_idx_roll == 0:
            v_curr = torch.tensor(v_train_init, dtype=torch.float32, device=device)
        else:
            v_curr = (y_train_t[start_idx_roll] - y_train_t[start_idx_roll-1]) / dt
            
        pred_y_list = []
        for i in range(window_len):
            u_val = u_train_t[start_idx_roll + i]
            y_curr, v_curr = rk4_step(y_curr, v_curr, u_val, model, dt)
            pred_y_list.append(y_curr)
            
        pred_y_tensor = torch.stack(pred_y_list)
        true_y_tensor = y_train_t[start_idx_roll + 1 : start_idx_roll + 1 + window_len]
        
        loss = torch.mean((pred_y_tensor - true_y_tensor)**2)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

    # 5. Full-length Free-Run Test
    model.eval()
    print("Running full free-run integration...")
    
    with torch.no_grad():
        y_free = np.zeros(length)
        v_free = np.zeros(length)
        
        # Initial conditions at t=0 (index 0 of excerpt)
        y_curr = y_centered[0]
        v_curr = (y_centered[1] - y_centered[0]) / dt
        
        y_free[0] = y_curr
        v_free[0] = v_curr
        
        y_curr_t = torch.tensor(y_curr, dtype=torch.float32, device=device)
        v_curr_t = torch.tensor(v_curr, dtype=torch.float32, device=device)
        
        for i in range(length - 1):
            u_val = u_full_t[i]
            y_curr_t, v_curr_t = rk4_step(y_curr_t, v_curr_t, u_val, model, dt)
            y_free[i+1] = y_curr_t.item()
            v_free[i+1] = v_curr_t.item()
            
    # Compute RMSEs
    y_train_pred = y_free[:train_len]
    y_test_pred = y_free[train_len:]
    
    train_rmse = np.sqrt(np.mean((y_train_pred - y_train)**2))
    test_rmse = np.sqrt(np.mean((y_test_pred - y_test)**2))
    
    print(f"FREE-RUN TRAIN RMSE: {train_rmse:.6f}")
    print(f"FREE-RUN TEST RMSE: {test_rmse:.6f}")
    
    # 6. Plotting test segment comparison
    plt.figure(figsize=(10, 4))
    time_axis = np.arange(test_len) * dt
    plt.plot(time_axis, y_test, label="Measured y(t)", color="black", alpha=0.7)
    plt.plot(time_axis, y_test_pred, label="Free-Run Predicted y(t)", color="red", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title(f"Silverbox Test Segment Free-Run (RMSE: {test_rmse:.4f} V)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("silverbox_test_free_run.png")
    
if __name__ == "__main__":
    main()
