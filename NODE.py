#!/usr/bin/env python3
"""
Neural ODE on real Aggregated Hourly Charging Load (AHCL) from Dataset1_charging_reports.csv.
Everything is organized into clear, sequential functions:
  1) Load real session-level EV charging data
  2) Aggregate sessions to hourly AHCL and visualize
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




def aggregate_by_day_of_week(df_sessions):
    """Aggregate to hourly AHCL for each day of week."""
    df_sessions = df_sessions.copy()
    df_sessions["plugin_time"] = pd.to_datetime(df_sessions["plugin_time"])
    df_sessions["plugout_time"] = pd.to_datetime(df_sessions["plugout_time"])
    df_sessions["day_of_week"] = df_sessions["plugin_time"].dt.dayofweek
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    results = {}
    for day_idx, day_name in enumerate(day_names):
        df_day = df_sessions[df_sessions["day_of_week"] == day_idx]
        if len(df_day) == 0:
            continue
        hours = np.arange(24, dtype=float)
        ahcl = np.zeros(24, dtype=float)
        for s, e, energy in zip(df_day["plugin_time"], df_day["plugout_time"], df_day["energy_session"]):
            s_sec = s.hour * 3600 + s.minute * 60 + s.second
            e_sec = e.hour * 3600 + e.minute * 60 + e.second
            if e_sec < s_sec:
                e_sec += 24 * 3600
            start_hour, end_hour = int(s_sec // 3600), int(e_sec // 3600)
            total_s = e_sec - s_sec
            if total_s <= 0:
                continue
            for h in range(start_hour, end_hour + 1):
                h_display = h % 24
                overlap_start = max(s_sec, h * 3600)
                overlap_end = min(e_sec, (h + 1) * 3600)
                overlap_s = max(0.0, overlap_end - overlap_start)
                ahcl[h_display] += energy * (overlap_s / total_s)
        df_hourly = pd.DataFrame({"hour": hours.astype(int), "ahcl_kwh_per_hour": ahcl})
        results[day_name] = (hours, ahcl, df_hourly)
        print(f"  {day_name:12s}: {len(df_day):5d} sessions → peak {ahcl.max():.2f} kWh/h")
    return results
def aggregate_by_season(df_sessions):
    """Aggregate to hourly AHCL for each season."""
    df_sessions = df_sessions.copy()
    df_sessions["plugin_time"] = pd.to_datetime(df_sessions["plugin_time"])
    df_sessions["plugout_time"] = pd.to_datetime(df_sessions["plugout_time"])
    df_sessions["month"] = df_sessions["plugin_time"].dt.month
    def get_season(month):
        if month in [12, 1, 2]: return "Winter"
        elif month in [3, 4, 5]: return "Spring"
        elif month in [6, 7, 8]: return "Summer"
        else: return "Fall"
    df_sessions["season"] = df_sessions["month"].apply(get_season)
    results = {}
    for season_name in ["Winter", "Spring", "Summer", "Fall"]:
        df_season = df_sessions[df_sessions["season"] == season_name]
        if len(df_season) == 0:
            continue
        hours = np.arange(24, dtype=float)
        ahcl = np.zeros(24, dtype=float)
        for s, e, energy in zip(df_season["plugin_time"], df_season["plugout_time"], df_season["energy_session"]):
            s_sec = s.hour * 3600 + s.minute * 60 + s.second
            e_sec = e.hour * 3600 + e.minute * 60 + e.second
            if e_sec < s_sec:
                e_sec += 24 * 3600
            start_hour, end_hour = int(s_sec // 3600), int(e_sec // 3600)
            total_s = e_sec - s_sec
            if total_s <= 0:
                continue
            for h in range(start_hour, end_hour + 1):
                h_display = h % 24
                overlap_start = max(s_sec, h * 3600)
                overlap_end = min(e_sec, (h + 1) * 3600)
                overlap_s = max(0.0, overlap_end - overlap_start)
                ahcl[h_display] += energy * (overlap_s / total_s)
        df_hourly = pd.DataFrame({"hour": hours.astype(int), "ahcl_kwh_per_hour": ahcl})
        results[season_name] = (hours, ahcl, df_hourly)
        print(f"  {season_name:12s}: {len(df_season):5d} sessions → peak {ahcl.max():.2f} kWh/h")
    return results
def aggregate_by_weekday_weekend(df_sessions):
    """Aggregate to hourly AHCL for weekday vs weekend."""
    df_sessions = df_sessions.copy()
    df_sessions["plugin_time"] = pd.to_datetime(df_sessions["plugin_time"])
    df_sessions["plugout_time"] = pd.to_datetime(df_sessions["plugout_time"])
    df_sessions["day_of_week"] = df_sessions["plugin_time"].dt.dayofweek
    df_sessions["is_weekend"] = df_sessions["day_of_week"] >= 5
    results = {}
    for is_weekend, label in [(False, "Weekday"), (True, "Weekend")]:
        df_type = df_sessions[df_sessions["is_weekend"] == is_weekend]
        if len(df_type) == 0:
            continue
        hours = np.arange(24, dtype=float)
        ahcl = np.zeros(24, dtype=float)
        for s, e, energy in zip(df_type["plugin_time"], df_type["plugout_time"], df_type["energy_session"]):
            s_sec = s.hour * 3600 + s.minute * 60 + s.second
            e_sec = e.hour * 3600 + e.minute * 60 + e.second
            if e_sec < s_sec:
                e_sec += 24 * 3600
            start_hour, end_hour = int(s_sec // 3600), int(e_sec // 3600)
            total_s = e_sec - s_sec
            if total_s <= 0:
                continue
            for h in range(start_hour, end_hour + 1):
                h_display = h % 24
                overlap_start = max(s_sec, h * 3600)
                overlap_end = min(e_sec, (h + 1) * 3600)
                overlap_s = max(0.0, overlap_end - overlap_start)
                ahcl[h_display] += energy * (overlap_s / total_s)
        df_hourly = pd.DataFrame({"hour": hours.astype(int), "ahcl_kwh_per_hour": ahcl})
        results[label] = (hours, ahcl, df_hourly)
        print(f"  {label:12s}: {len(df_type):5d} sessions → peak {ahcl.max():.2f} kWh/h")
    return results

def load_real_data(csv_path: str = "Dataset1_charging_reports.csv"):
    """
    Load real EV charging session data from CSV with semicolon delimiters and decimal commas.
    Expected columns: location, user_id, session_id, plugin_time, plugout_time, connection_time, energy_session

    Returns:
        tuple: (DataFrame with data, Path to CSV)
    """
    # Load with semicolon delimiter and decimal comma
    df = pd.read_csv(csv_path, sep=";", decimal=",")
    
    # Filter data to only include sessions from location "ASK"
    df = df[df["location"] == "ASK"]
    
    # Strip quotes from column names and values
    df.columns = df.columns.str.strip('"')
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip('"')
    
    # Convert plugin_time and plugout_time to datetime
    df["plugin_time"] = pd.to_datetime(df["plugin_time"])
    df["plugout_time"] = pd.to_datetime(df["plugout_time"])
    
    # Ensure energy_session is numeric
    df["energy_session"] = pd.to_numeric(df["energy_session"], errors="coerce")
    
    # Remove rows with missing critical values
    df = df.dropna(subset=["plugin_time", "plugout_time", "energy_session"])
    
    print(f"Loaded {len(df)} charging sessions from {csv_path}")
    print(f"Date range: {df['plugin_time'].min()} to {df['plugout_time'].max()}")
    
    return df, Path(csv_path)
def load_and_prepare_data():
    print("Loading real data...")
    df, csv_path = load_real_data()

    print("\n" + "="*60)
    print("Single Neural ODE with Day-of-Week Aggregation")
    print("="*60)

    # Aggregate by day-of-week
    day_profiles = aggregate_by_day_of_week(df)
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Combine into a 168-hour vector
    ahcl_combined = np.concatenate([day_profiles[day][1] for day in day_names])
    hours_combined = np.arange(168, dtype=float)

    print(f"\nCombined AHCL shape: {ahcl_combined.shape}")
    print(f"Peak hourly load across all days: {ahcl_combined.max():.2f} kWh/h")
    print(f"Min hourly load: {ahcl_combined.min():.2f} kWh/h")

    return df, ahcl_combined, hours_combined
def build_features():
    print("\nDefining features for Neural ODE...")
    features = torch.zeros((168, 3), dtype=torch.float32)

    for t in range(168):
        day_idx = t // 24
        hour_in_day = t % 24

        features[t, 0] = np.sin(2 * np.pi * hour_in_day / 24)
        features[t, 1] = np.cos(2 * np.pi * hour_in_day / 24)
        features[t, 2] = day_idx / 7.0

    return features
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
class NeuralODEHourly(nn.Module):
    def __init__(self, hidden=32, dt=1.0, y0_init=None):
        super().__init__()
        self.func = ODEFunc(hidden)
        self.y0 = nn.Parameter(torch.tensor([[2.0]], dtype=torch.float32))  # Initialize with baseline value
        self.dt = dt

    def forward(self, U):
        return euler_integrate(self.func, self.y0, U, self.dt)
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
def build_model():
    print("Initializing Neural ODE model...")

    model = NeuralODEHourly(hidden=32, dt=1.0)
    model.func.net = nn.Sequential(
        nn.Linear(1 + 3, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    return model, optimizer
def train_model(model, optimizer, features, y_true, epochs=2000):
    print("\nTraining model (on normalized data [0,1])...")
    print(f"  Target range: [{y_true.min():.4f}, {y_true.max():.4f}]")
    losses = []
    best_loss = float('inf')

    for epoch in range(epochs):
        optimizer.zero_grad()
        y_hat = model(features)
        loss = F.mse_loss(y_hat, y_true)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        best_loss = min(best_loss, loss.item())

        if (epoch + 1) % 200 == 0:
            print(f"  epoch {epoch+1:4d} | MSE {loss.item():.6f}")

    print(f"\n  Final MSE (normalized): {losses[-1]:.6f}")
    print(f"  Best MSE (normalized): {best_loss:.6f}")
    
    with torch.no_grad():
        y_pred = model(features).cpu().numpy()

    return y_pred, losses
def compute_and_save_metrics(output_base, y_true_np, y_pred):
    print("\nSaving results...")

    mse = np.mean((y_true_np - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true_np - y_pred))
    r2 = 1.0 - (np.sum((y_true_np - y_pred) ** 2) /
                np.sum((y_true_np - np.mean(y_true_np)) ** 2))

    print("\nPerformance Metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")

    metrics = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

    (output_base / "metrics.txt").write_text(
        "Single Neural ODE with Day-of-Week Aggregation (168 hours)\n\n" +
        "\n".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
    )

    return metrics
def plot_simple_results(output_base, hours, y_true, y_pred, losses):
    """Simple plotting function for 168-hour predictions."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Predictions vs true
    axes[0].plot(hours, y_true, "o-", label="True AHCL", alpha=0.7, markersize=4)
    axes[0].plot(hours, y_pred, "s--", label="Predicted AHCL", alpha=0.7, markersize=3)
    axes[0].axvline(24, color="gray", linestyle=":", alpha=0.5, label="Day boundaries")
    for d in range(1, 7):
        axes[0].axvline(d * 24, color="gray", linestyle=":", alpha=0.5)
    axes[0].set_ylabel("kWh/h")
    axes[0].set_title("168-Hour Forecast: Day-of-Week Aggregation (Normalized Training)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss curve
    axes[1].plot(losses, linewidth=1.5, color="C1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE Loss (normalized)")
    axes[1].set_title("Training Loss Trajectory")
    axes[1].grid(True, alpha=0.3)
    
    fit_path = output_base / "ahcl_fit_168h.png"
    plt.tight_layout()
    plt.savefig(fit_path, dpi=100)
    print(f"  Saved fit plot: {fit_path.resolve()}")
    plt.close()


def save_model_state(model, path: Path):
    """Save model state_dict to the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(path))
    print(f"  Saved model state to: {path.resolve()}")


def load_model_state(path: Path):
    """Instantiate a model and load state_dict from path."""
    model, _ = build_model()
    state = torch.load(str(path), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    print(f"  Loaded model state from: {path.resolve()}")
    return model


def predict_and_save_samples(model, features, ahcl_original, y_min, y_max, sample_indices=None, output_base=Path("outputs/new_perf")):
    """
    Run the model on `features`, denormalize predictions, save a small CSV with chosen samples.
    Returns a DataFrame of the samples.
    """
    if sample_indices is None:
        sample_indices = [0, 6, 12, 18, 23, 24, 48, 72, 120, 167]

    with torch.no_grad():
        y_pred_norm = model(features).cpu().numpy().reshape(-1)

    # Denormalize
    y_pred = y_pred_norm * (y_max - y_min) + y_min

    rows = []
    for idx in sample_indices:
        if idx < 0 or idx >= len(ahcl_original):
            continue
        day = int(idx // 24)
        hour = int(idx % 24)
        true_v = float(ahcl_original[idx])
        pred_v = float(y_pred[idx])
        err = true_v - pred_v
        rows.append({"index": idx, "day": day, "hour": hour, "true_kwh_h": true_v, "pred_kwh_h": pred_v, "error": err})

    df_samples = pd.DataFrame(rows)
    output_base.mkdir(parents=True, exist_ok=True)
    csv_path = output_base / "sample_predictions.csv"
    df_samples.to_csv(csv_path, index=False)
    print(f"  Saved sample predictions to: {csv_path.resolve()}")

    print("\nSample predictions (true vs pred):")
    print(df_samples.to_string(index=False, float_format='{:0.3f}'.format))

    return df_samples


def main():
    # Create output directory
    output_base = Path("outputs/new_perf")
    output_base.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    df, ahcl_combined, hours_combined = load_and_prepare_data()
    features = build_features()
    
    # NORMALIZE targets to [0, 1] for stable training
    y_min, y_max = ahcl_combined.min(), ahcl_combined.max()
    y_normalized = (ahcl_combined - y_min) / (y_max - y_min + 1e-8)
    y_true = torch.tensor(y_normalized, dtype=torch.float32)
    print(f"\nNormalization: min={y_min:.2f} kWh/h, max={y_max:.2f} kWh/h")

    # Build and train the model
    model, optimizer = build_model()
    y_pred_norm, losses = train_model(model, optimizer, features, y_true)
    
    # DENORMALIZE predictions back to original scale
    y_pred = y_pred_norm * (y_max - y_min) + y_min
    print(f"\nDenormalized prediction range: [{y_pred.min():.2f}, {y_pred.max():.2f}] kWh/h")

    # Save metrics and plots (using original scale)
    compute_and_save_metrics(output_base, ahcl_combined, y_pred)
    plot_simple_results(output_base, hours_combined, ahcl_combined, y_pred, losses)

    # Save the trained model state and demonstrate loading + prediction on samples
    model_path = output_base / "neural_ode_state.pth"
    save_model_state(model, model_path)

    # Load into a fresh model instance and predict
    loaded_model = load_model_state(model_path)

    # Produce and save a small table of handpicked sample predictions
    predict_and_save_samples(loaded_model, features, ahcl_combined, y_min, y_max, sample_indices=None, output_base=output_base)


if __name__ == "__main__":
    main()