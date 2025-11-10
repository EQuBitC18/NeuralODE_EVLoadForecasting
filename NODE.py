#!/usr/bin/env python3
"""
Minimal Neural ODE on toy Aggregated Hourly Charging Load (AHCL).
Everything is organized into clear, sequential functions:
  1) Build toy 24h AHCL data
  2) Visualize the data
  3) Define a tiny Neural ODE (dy/dt = f_theta(y, hour-features))
  4) Train with explicit Euler integration
  5) Plot forecast vs. truth + loss curve
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

class ODEFunc(nn.Module):
    """dy/dt = f_theta(y, u) with a tiny MLP."""
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + 2, hidden),  # input: y (1) + time features (2)
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # Xavier init
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, y, u):
        return self.net(torch.cat([y, u], dim=-1)) 
def euler_integrate(func, y0, U, dt=1.0):
    """
    Explicit Euler integrator over a 24-step grid (1-hour steps).
    y0: [1,1]
    U:  [T,2]
    returns: y trajectory [T]
    """
    y = y0.clone()
    ys = []
    for t in range(U.shape[0]):
        dy = func(y, U[t:t+1, :])      # [1,1]
        y = y + dy * dt                # Euler step
        ys.append(y.squeeze(0).squeeze(-1))
    return torch.stack(ys, dim=0)      # [T]
class NeuralODEHourly(nn.Module):
    def __init__(self, hidden=32, dt=1.0, y0_init=None):
        super().__init__()
        self.func = ODEFunc(hidden)
        self.y0 = nn.Parameter(torch.tensor([[2.0]], dtype=torch.float32))  # Initialize with baseline value
        self.dt = dt

    def forward(self, U):
        return euler_integrate(self.func, self.y0, U, self.dt)
def create_toy_data(
    n_sessions: int = 120,
    base_date: str = "2025-01-01",
    out_dir: str = "outputs",
):
    """
    Create toy EV charging session data with four columns:
        - plugin_time:   "YYYY-MM-DD HH:MM:SS"
        - plugout_time:  "YYYY-MM-DD HH:MM:SS"
        - connection_time: "HH:MM:SS" (duration)
        - energy_session: float in [1.00, 100.00]

    Returns:
        tuple: (DataFrame with data, Path to CSV)
    """
    from pathlib import Path
    from datetime import datetime, timedelta
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)

    # --- parameters for session sampling ---
    day_start = datetime.strptime(base_date + " 00:00:00", "%Y-%m-%d %H:%M:%S")
    day_end = day_start + timedelta(days=1) - timedelta(seconds=1)

    # durations between 10 minutes and 5 hours (in seconds)
    min_dur_s = 10 * 60
    max_dur_s = 5 * 60 * 60

    plugin_secs = rng.integers(0, 24 * 60 * 60 - min_dur_s, size=n_sessions)
    durations_s = rng.integers(min_dur_s, max_dur_s + 1, size=n_sessions)

    plugin_times = []
    plugout_times = []
    connection_times = []
    energies = []

    for start_s, dur_s in zip(plugin_secs, durations_s):
        plug_in = day_start + timedelta(seconds=int(start_s))
        plug_out_raw = plug_in + timedelta(seconds=int(dur_s))
        plug_out = min(plug_out_raw, day_end)

        conn_delta = plug_out - plug_in
        conn_s = int(conn_delta.total_seconds())

        plugin_times.append(plug_in.strftime("%Y-%m-%d %H:%M:%S"))
        plugout_times.append(plug_out.strftime("%Y-%m-%d %H:%M:%S"))

        hh = conn_s // 3600
        mm = (conn_s % 3600) // 60
        ss = conn_s % 60
        connection_times.append(f"{hh:02d}:{mm:02d}:{ss:02d}")

        power_kw = rng.uniform(3.0, 11.0)
        noise = rng.normal(0.0, 2.0)
        energy = (conn_s / 3600.0) * power_kw + noise
        energy = float(np.clip(energy, 1.0, 100.0))
        energies.append(round(energy, 2))

    df = pd.DataFrame(
        {
            "plugin_time": plugin_times,
            "plugout_time": plugout_times,
            "connection_time": connection_times,
            "energy_session": energies,
        }
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "toy_sessions.csv"
    df.to_csv(csv_path, index=False)

    return df, csv_path
def aggregate_sessions_to_hourly(df_sessions, base_date=None):
    """
    Aggregate session-level data to hourly AHCL (kWh per hour) over a single day.

    Args:
        df_sessions: DataFrame with columns ['plugin_time','plugout_time','energy_session']
        base_date: optional date string 'YYYY-MM-DD' to anchor the 24h period; if None, infer from plugin_time

    Returns:
        hours (np.ndarray[24]), ahcl (np.ndarray[24]), df_hourly (pd.DataFrame)
    """
    from datetime import datetime, timedelta

    # parse times
    plugin = pd.to_datetime(df_sessions["plugin_time"])
    plugout = pd.to_datetime(df_sessions["plugout_time"])

    if base_date is None:
        base_date = plugin.dt.date.min().strftime("%Y-%m-%d")

    day_start = datetime.strptime(base_date + " 00:00:00", "%Y-%m-%d %H:%M:%S")
    hours = np.arange(24, dtype=float)
    ahcl = np.zeros(24, dtype=float)

    # For each session, distribute its energy across overlapping hours proportional to overlap seconds
    for s, e, energy in zip(plugin, plugout, df_sessions["energy_session"]):
        # clip to day
        s_clipped = max(s.to_pydatetime(), day_start)
        e_clipped = min(e.to_pydatetime(), day_start + timedelta(days=1))
        if e_clipped <= s_clipped:
            continue
        total_s = (e_clipped - s_clipped).total_seconds()
        if total_s <= 0:
            continue
        # iterate overlapping hours
        start_hour = int((s_clipped - day_start).total_seconds() // 3600)
        end_hour = int((e_clipped - day_start).total_seconds() // 3600)
        # clamp
        start_hour = max(0, min(23, start_hour))
        end_hour = max(0, min(23, end_hour))

        for h in range(start_hour, end_hour + 1):
            hour_start = day_start + timedelta(hours=h)
            hour_end = hour_start + timedelta(hours=1)
            overlap_start = max(s_clipped, hour_start)
            overlap_end = min(e_clipped, hour_end)
            overlap_s = max(0.0, (overlap_end - overlap_start).total_seconds())
            ahcl[h] += energy * (overlap_s / total_s)

    df_hourly = pd.DataFrame({"hour": hours.astype(int), "ahcl_kwh_per_hour": ahcl})
    return hours, ahcl, df_hourly
def visualize_data(df):
    """
    2) Visualize the toy data
    Args:
        df: DataFrame containing hour and ahcl_kwh_per_hour data
    Returns:
        Path: Path to the saved plot
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["hour"].values, df["ahcl_kwh_per_hour"].values, marker="o")
    ax.set_title("Toy Aggregated Hourly Charging Load (AHCL) — 24h")
    ax.set_xlabel("Hour of day (0–23)")
    ax.set_ylabel("kWh per hour")
    ax.grid(True, alpha=0.3)

    peak_idx = int(np.argmax(df["ahcl_kwh_per_hour"].values))
    peak_val = float(df["ahcl_kwh_per_hour"].iloc[peak_idx])
    ax.annotate(
        f"Peak @ {peak_idx}:00",
        xy=(peak_idx, peak_val),
        xytext=(peak_idx, peak_val + 0.6),
        arrowprops=dict(arrowstyle="->"),
    )

    out = Path(".")
    plot_data_path = out / "outputs/ahcl_toy_plot.png"
    fig.tight_layout()
    fig.savefig(plot_data_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    
    return plot_data_path
def define_neural_ode(ahcl):
    """
    3) Define Neural ODE components (tiny MLP)
    Args:
        ahcl: numpy array with AHCL data
    Returns:
        tuple: (feature tensor, true values tensor)
    """
    torch.manual_seed(0)

    # Features: hour-of-day encoded as sin/cos (periodic)
    hours_t = torch.arange(24, dtype=torch.float32)
    feat = torch.stack(
        [
            torch.sin(2.0 * torch.pi * hours_t / 24.0),
            torch.cos(2.0 * torch.pi * hours_t / 24.0),
        ],
        dim=1,
    )  # [24, 2]
    y_true = torch.tensor(ahcl, dtype=torch.float32)  # [24]
    
    return feat, y_true
def train_model(feat, y_true):
    """
    4) Train the Neural ODE on the toy series (MSE)
    Args:
        feat: feature tensor [24, 2]
        y_true: true values tensor [24]
    Returns:
        tuple: (predictions numpy array, list of losses)
    """
    model = NeuralODEHourly(hidden=32, dt=1.0)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)  # simple optimizer

    epochs = 2000
    losses = []
    for epoch in range(epochs):
        opt.zero_grad()
        y_hat = model(feat)                     # [24]
        loss = F.mse_loss(y_hat, y_true)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if (epoch + 1) % 200 == 0:
            print(f"epoch {epoch+1:4d} | MSE {loss.item():.4f}")

    # Final predictions
    with torch.no_grad():
        y_pred = model(feat).cpu().numpy()
        
    return y_pred, losses
def plot_results(hours, ahcl, y_pred, losses, csv_path, plot_data_path):
    """
    5) Plot the fit and training loss
    Args:
        hours: numpy array with hours
        ahcl: numpy array with true AHCL values
        y_pred: numpy array with predicted values
        losses: list of training losses
        csv_path: Path object for the data CSV
        plot_data_path: Path object for the data plot
    """
    out = Path(".")
    # Fit vs truth
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(hours, ahcl, marker="o", label="Truth (toy AHCL)")
    ax1.plot(hours, y_pred, marker="s", linestyle="--", label="Neural ODE forecast")
    ax1.set_title("Neural ODE fit on toy AHCL")
    ax1.set_xlabel("Hour of day (0–23)")
    ax1.set_ylabel("kWh per hour")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fit_path = out / "outputs/ahcl_toy_fit.png"
    fig1.tight_layout()
    fig1.savefig(fit_path, dpi=160, bbox_inches="tight")
    plt.close(fig1)

    # Loss curve
    fig2, ax2 = plt.subplots(figsize=(10, 3.2))
    ax2.plot(np.arange(1, len(losses) + 1), losses)
    ax2.set_title("Training loss (MSE)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE")
    ax2.grid(True, alpha=0.3)
    loss_path = out / "outputs/ahcl_toy_loss.png"
    fig2.tight_layout()
    fig2.savefig(loss_path, dpi=160, bbox_inches="tight")
    plt.close(fig2)

    # Final prints
    print(f"Saved data CSV to: {csv_path.resolve()}")
    print(f"Saved toy data plot to: {plot_data_path.resolve()}")
    print(f"Saved fit plot to: {fit_path.resolve()}")
    print(f"Saved loss plot to: {loss_path.resolve()}")

def main():
    # 1) Create toy session-level data
    df, csv_path = create_toy_data()
    print(df.head())

    # 2) Aggregate sessions to hourly AHCL and visualize
    hours, ahcl, df_hourly = aggregate_sessions_to_hourly(df)
    plot_data_path = visualize_data(df_hourly)

    # 3) Define Neural ODE components from aggregated AHCL
    feat, y_true = define_neural_ode(ahcl)

    # 4) Train the model
    y_pred, losses = train_model(feat, y_true)

    # 5) Plot the results
    plot_results(hours, ahcl, y_pred, losses, csv_path, plot_data_path)

if __name__ == "__main__":
    main()