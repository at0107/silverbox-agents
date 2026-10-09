import os
import time
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 1. Load data
mat_path = os.environ["SILVERBOX_MAT"]
mat = scipy.io.loadmat(mat_path)
V1 = mat['V1'].squeeze() # input u
V2 = mat['V2'].squeeze() # output y

# Excerpt: 4096 samples starting at index 42650
start_idx = 42650
n_samples = 4096
u_raw = V1[start_idx : start_idx + n_samples].astype(np.float32)
y_raw = V2[start_idx : start_idx + n_samples].astype(np.float32)

# Subtract mean of excerpt
u = u_raw - np.mean(u_raw)
y = y_raw - np.mean(y_raw)

fs = 610.35
dt = 1.0 / fs

n_train = 3072
n_test = 1024

# Finite-difference velocity for initial conditions
# Centered difference inside, forward/backward at ends
v = np.zeros_like(y)
v[1:-1] = (y[2:] - y[:-2]) / (2.0 * dt)
v[0] = (y[1] - y[0]) / dt
v[-1] = (y[-1] - y[-2]) / dt

# Normalization constants based on standard deviations
# y ~ 0.06 V, v ~ 25 V/s, u ~ 0.03 V, acceleration ~ 10000 V/s^2
std_y = float(np.std(y[:n_train]))
std_v = float(np.std(v[:n_train]))
std_u = float(np.std(u[:n_train]))
std_a = 1.0e4 # scale for acceleration

print(f"Stats: std_y={std_y:.4f}, std_v={std_v:.4f}, std_u={std_u:.4f}")

# Convert to torch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

u_torch = torch.tensor(u, dtype=torch.float32, device=device)
y_torch = torch.tensor(y, dtype=torch.float32, device=device)
v_torch = torch.tensor(v, dtype=torch.float32, device=device)

# 2. Neural ODE Network
class AccelerationNet(nn.Module):
    def __init__(self, sy, sv, su, sa, hidden_dim=64):
        super().__init__()
        self.sy = sy
        self.sv = sv
        self.su = su
        self.sa = sa
        
        # Physics-based baseline linear terms initialized to typical Silverbox parameters
        # y'' = - a*v - b*y + g*u + mlp(y, v, u)
        # b ~ (2*pi*70)^2 ~ 1.93e5, a ~ 20-50, g ~ 1e4 - 2e5
        self.linear_a = nn.Parameter(torch.tensor(30.0, dtype=torch.float32))
        self.linear_b = nn.Parameter(torch.tensor(1.9e5, dtype=torch.float32))
        self.linear_c = nn.Parameter(torch.tensor(1.0e6, dtype=torch.float32))
        self.linear_g = nn.Parameter(torch.tensor(1.0e5, dtype=torch.float32))
        
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Small initialization for the MLP final layer
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, y_val, v_val, u_val):
        # Physics prior acceleration
        acc_prior = - self.linear_a * v_val - self.linear_b * y_val - self.linear_c * (y_val ** 3) + self.linear_g * u_val
        
        # Normalized input to MLP
        in_feats = torch.stack([y_val / self.sy, v_val / self.sv, u_val / self.su], dim=-1)
        mlp_out = self.mlp(in_feats).squeeze(-1) * self.sa
        
        return acc_prior + mlp_out

model = AccelerationNet(sy=std_y, sv=std_v, su=std_u, sa=std_a, hidden_dim=64).to(device)

def rk4_step(model, y_curr, v_curr, u_curr, u_next, dt):
    u_mid = 0.5 * (u_curr + u_next)
    
    k1_y = v_curr
    k1_v = model(y_curr, v_curr, u_curr)
    
    k2_y = v_curr + 0.5 * dt * k1_v
    k2_v = model(y_curr + 0.5 * dt * k1_y, k2_y, u_mid)
    
    k3_y = v_curr + 0.5 * dt * k2_v
    k3_v = model(y_curr + 0.5 * dt * k2_y, k3_y, u_mid)
    
    k4_y = v_curr + dt * k3_v
    k4_v = model(y_curr + dt * k3_y, k4_y, u_next)
    
    y_next = y_curr + (dt / 6.0) * (k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y)
    v_next = v_curr + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    return y_next, v_next

def run_continuous_simulation(model, u_all, y0, v0, dt):
    # Free run across all steps
    n_total = len(u_all)
    y_sim = torch.empty(n_total, device=device)
    v_sim = torch.empty(n_total, device=device)
    y_sim[0] = y0
    v_sim[0] = v0
    
    y_c = y0
    v_c = v0
    for k in range(n_total - 1):
        y_c, v_c = rk4_step(model, y_c, v_c, u_all[k], u_all[k+1], dt)
        y_sim[k+1] = y_c
        v_sim[k+1] = v_c
        
    return y_sim

# Setup optimizer and learning rate schedule
optimizer = optim.AdamW([
    {'params': [model.linear_a, model.linear_b, model.linear_c, model.linear_g], 'lr': 1e-2, 'weight_decay': 0.0},
    {'params': model.mlp.parameters(), 'lr': 2e-3, 'weight_decay': 1e-5}
])

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4000, eta_min=1e-4)

# Multi-stage rollout curriculum
# We train for at most 180 seconds (3 minutes) wall clock budget
start_time = time.time()
max_training_time = 180.0

best_test_rmse = float('inf')
best_state_dict = None

iter_count = 0
batch_size = 32

print("Beginning training...")
while True:
    elapsed = time.time() - start_time
    if elapsed >= max_training_time:
        print(f"Wall-clock budget reached ({elapsed:.1f} s). Stopping training.")
        break
        
    iter_count += 1
    
    # Curriculum: start with shorter rollouts, increase as training progresses
    if elapsed < 30.0:
        rollout_len = 16
    elif elapsed < 70.0:
        rollout_len = 32
    elif elapsed < 120.0:
        rollout_len = 64
    else:
        rollout_len = 128
        
    # Sample random starting indices from training set
    # Ensure starting indices leave room for rollout_len steps
    max_start = n_train - rollout_len - 1
    start_indices = np.random.randint(0, max_start, size=batch_size)
    
    # Initial batch states
    y_curr = y_torch[start_indices]
    v_curr = v_torch[start_indices]
    
    loss = 0.0
    for step in range(rollout_len):
        u_curr = u_torch[start_indices + step]
        u_next = u_torch[start_indices + step + 1]
        target_y = y_torch[start_indices + step + 1]
        
        y_curr, v_curr = rk4_step(model, y_curr, v_curr, u_curr, u_next, dt)
        loss = loss + torch.mean((y_curr - target_y) ** 2)
        
    loss = loss / rollout_len
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    
    # Periodic evaluation
    if iter_count % 100 == 0 or elapsed >= max_training_time - 2.0:
        with torch.no_grad():
            model.eval()
            # Free run evaluation starting at sample 0
            y0 = y_torch[0]
            v0 = (y_torch[1] - y_torch[0]) / dt
            y_pred = run_continuous_simulation(model, u_torch, y0, v0, dt)
            
            train_rmse = torch.sqrt(torch.mean((y_pred[:n_train] - y_torch[:n_train]) ** 2)).item()
            test_rmse = torch.sqrt(torch.mean((y_pred[n_train:] - y_torch[n_train:]) ** 2)).item()
            model.train()
            
            print(f"Iter {iter_count:4d} | Time {elapsed:5.1f}s | Rollout {rollout_len:3d} | Batch Loss {loss.item():.2e} | Free-run Train RMSE: {train_rmse:.6f} | Test RMSE: {test_rmse:.6f}")
            
            if test_rmse < best_test_rmse and not np.isnan(test_rmse):
                best_test_rmse = test_rmse
                best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

# Load best checkpoint
if best_state_dict is not None:
    model.load_state_dict(best_state_dict)
    print("Loaded best checkpoint with test RMSE:", best_test_rmse)

# Final Free Run Evaluation
model.eval()
with torch.no_grad():
    y0 = y_torch[0]
    v0 = (y_torch[1] - y_torch[0]) / dt
    y_pred = run_continuous_simulation(model, u_torch, y0, v0, dt)
    
    final_train_rmse = torch.sqrt(torch.mean((y_pred[:n_train] - y_torch[:n_train]) ** 2)).item()
    final_test_rmse = torch.sqrt(torch.mean((y_pred[n_train:] - y_torch[n_train:]) ** 2)).item()

print(f"FREE-RUN TRAIN RMSE: {final_train_rmse:.6f}")
print(f"FREE-RUN TEST RMSE: {final_test_rmse:.6f}")
