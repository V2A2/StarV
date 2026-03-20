"""
RNN training for LIMO simulation
Qing Liu, 3/6/2025
"""
import random
from scipy.io import loadmat
import os
from pathlib import Path
from typing import Optional
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader,TensorDataset
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import mean_squared_error


pd.set_option("display.float_format", "{:.4f}".format)


def make_target_scaler_from_input_scaler(x_scaler, target_indices: list):
    if isinstance(x_scaler, MinMaxScaler):
        y_scaler = MinMaxScaler(feature_range=x_scaler.feature_range)
        y_scaler.min_ = x_scaler.min_[target_indices].copy()
        y_scaler.scale_ = x_scaler.scale_[target_indices].copy()
        y_scaler.data_min_ = x_scaler.data_min_[target_indices].copy()
        y_scaler.data_max_ = x_scaler.data_max_[target_indices].copy()
        y_scaler.data_range_ = x_scaler.data_range_[target_indices].copy()
        y_scaler.n_features_in_ = len(target_indices)
        y_scaler.n_samples_seen_ = x_scaler.n_samples_seen_
        return y_scaler

    if isinstance(x_scaler, StandardScaler):
        y_scaler = StandardScaler()
        y_scaler.mean_ = x_scaler.mean_[target_indices].copy()
        y_scaler.scale_ = x_scaler.scale_[target_indices].copy()
        y_scaler.var_ = x_scaler.var_[target_indices].copy()
        y_scaler.n_features_in_ = len(target_indices)
        y_scaler.n_samples_seen_ = x_scaler.n_samples_seen_
        return y_scaler

    raise RuntimeError(
        f"Unsupported scaler type for target extraction: {type(x_scaler)}. "
        "Use MinMaxScaler or StandardScaler."
    )


def build_windows(input_data: np.ndarray, target_data: np.ndarray, window_size: int):
    if input_data.shape[0] <= window_size:
        raise RuntimeError(
            f"Not enough steps ({input_data.shape[0]}) for window size {window_size}"
        )
    X = []
    y = []
    for start in range(input_data.shape[0] - window_size):
        end = start + window_size
        X.append(input_data[start:end, :])
        y.append(target_data[end, :])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# def prepare_eval_dataset_from_csv(
#     csv_path: str,
#     input_cols: list,
#     target_cols: list,
#     window_size: int,
#     x_scaler,
#     max_steps: Optional[int] = None,
#     save_processed_dir: Optional[str] = None,
# ):
#     df = pd.read_csv(csv_path)
#     required_cols = list(dict.fromkeys(input_cols + target_cols))
#     missing = [c for c in required_cols if c not in df.columns]
#     if missing:
#         raise RuntimeError(f"Missing required columns in {csv_path}: {missing}")

#     df = df.dropna(subset=required_cols).reset_index(drop=True)
#     if max_steps is not None:
#         df = df.iloc[:max_steps].reset_index(drop=True)

#     input_raw = df[input_cols].to_numpy(dtype=np.float32)
#     target_raw = df[target_cols].to_numpy(dtype=np.float32)
#     input_scaled = x_scaler.transform(input_raw).astype(np.float32)
#     target_indices = [input_cols.index(c) for c in target_cols]
#     target_scaled = input_scaled[:, target_indices].astype(np.float32)

#     processed_csv_path = None
#     if save_processed_dir is not None:
#         processed_dir_path = Path(save_processed_dir)
#         processed_dir_path.mkdir(parents=True, exist_ok=True)
#         processed_csv_path = processed_dir_path / f"{Path(csv_path).stem}_processed_4f.csv"
#         pd.DataFrame(input_scaled, columns=input_cols).to_csv(processed_csv_path, index=False)

#     X_eval, _ = build_windows(input_scaled, target_scaled, window_size)
#     y_eval_raw = target_raw[window_size:].astype(np.float32)
#     return X_eval, y_eval_raw, len(df), processed_csv_path

class RNN_trajectory(object):

    def __init__(self, traj_path, scaler, val_ratio, win_size, LIMO_INPUT_COLS, LIMO_TARGET_COLS):
        self.data_path = traj_path
        self.scaler = scaler
        self.val_ratio = val_ratio
        self.window_size = win_size
        self.LIMO_INPUT_COLS = LIMO_INPUT_COLS
        self.LIMO_TARGET_COLS = LIMO_TARGET_COLS

    def build_next_state_windows(self, input_data: np.ndarray, target_data: np.ndarray):
        if input_data.shape[0] <= self.window_size:
            raise RuntimeError(
                f"Not enough steps ({input_data.shape[0]}) for window size {self.window_size}"
            )
        X = []
        y = []
        for start in range(input_data.shape[0] - self.window_size):
            end = start + self.window_size
            X.append(input_data[start:end, :])
            y.append(target_data[end, :])  # next step state S(T+1)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def extract_limo_trajectory_df(self, df: pd.DataFrame):
        new_df = df.copy()
        before = len(new_df)
        required_cols = list(dict.fromkeys(self.LIMO_INPUT_COLS + self.LIMO_TARGET_COLS))
        new_df = new_df.dropna(subset=required_cols).reset_index(drop=True)
        removed = before - len(new_df)
        if removed > 0:
            print(f"Removed {removed} rows with NaN/invalid values in state columns.")
        return new_df

    def train_test_win_split(self, X_all: np.ndarray, y_all: np.ndarray):
        split_idx = int((1.0 - self.val_ratio) * len(X_all))
        if split_idx <= 0 or split_idx >= len(X_all):
            raise RuntimeError(
                f"Invalid split index {split_idx} for {len(X_all)} windows. ")
        X_train = X_all[:split_idx]
        y_train = y_all[:split_idx]
        X_val = X_all[split_idx:]
        y_val = y_all[split_idx:]
        return X_train, y_train, X_val, y_val, split_idx

    def process_limo_next_state_dataset(self, processed_traj_path=None):
        csv_path_obj = Path(self.data_path)
        df_raw = pd.read_csv(self.data_path)

        required_cols = list(dict.fromkeys(self.LIMO_INPUT_COLS + self.LIMO_TARGET_COLS))
        missing = [c for c in required_cols if c not in df_raw.columns]
        if missing:
            raise RuntimeError(f"Missing required columns in {self.data_path}: {missing}")

        df_new = self.extract_limo_trajectory_df(df_raw) # skip NaN and inf elements
        input_data_raw = df_new[self.LIMO_INPUT_COLS].to_numpy(dtype=np.float32)
        target_data_raw = df_new[self.LIMO_TARGET_COLS].to_numpy(dtype=np.float32)

        x_scaler = self.scaler if self.scaler is not None else StandardScaler()

        # 1) Scale full input trajectory first.
        input_data_scaled = x_scaler.fit_transform(input_data_raw).astype(np.float32)

        # 2) Extract scaled target trajectory directly from scaled input trajectory.
        target_indices = [self.LIMO_INPUT_COLS.index(c) for c in self.LIMO_TARGET_COLS]
        target_data_scaled = input_data_scaled[:, target_indices].astype(np.float32)
        y_scaler = make_target_scaler_from_input_scaler(x_scaler, target_indices)

        # 3) Save scaled full input trajectory as processed CSV.
        if processed_traj_path is None:
            processed_dir_path = (
                Path(os.path.dirname(os.path.abspath(__file__)))
                / "util/data/LIMO_trajectories/limo_processed"
            )
        else:
            processed_dir_path = Path(processed_traj_path)
        processed_dir_path.mkdir(parents=True, exist_ok=True)

        X_all_scaled_csv_path = processed_dir_path / f"{csv_path_obj.stem}_processed_4f.csv"
        pd.DataFrame(input_data_scaled, columns=self.LIMO_INPUT_COLS).to_csv(
            X_all_scaled_csv_path, index=False
        )

        # 4) Build windows from scaled trajectory with full next-state targets.
        X_all, y_all = self.build_next_state_windows(input_data_scaled, target_data_scaled)

        X_train_processed, y_train_processed, X_val_processed, y_val_processed, split_idx = (
            self.train_test_win_split(X_all, y_all)
        )

        # 5) Keep raw validation next-state targets and state ranges for evaluation.
        y_all_raw = target_data_raw[self.window_size:].astype(np.float32)
        y_val_raw = y_all_raw[split_idx:].astype(np.float32)
        state_range_raw = (
            np.max(target_data_raw, axis=0) - np.min(target_data_raw, axis=0)
        ).astype(np.float32)
        safe_state_range_raw = np.where(
            np.abs(state_range_raw) < 1e-8, 1.0, state_range_raw
        ).astype(np.float32)  # avoids divide-by-zero is range is an constant

        return {
            "df_raw": df_raw,
            "df_new": df_new,
            "input_cols": self.LIMO_INPUT_COLS,
            "target_cols": self.LIMO_TARGET_COLS,
            "X_train": X_train_processed.astype(np.float32),
            "y_train": y_train_processed.astype(np.float32),
            "X_val": X_val_processed.astype(np.float32),
            "y_val": y_val_processed.astype(np.float32),
            "y_val_raw": y_val_raw.astype(np.float32),
            "state_range_raw": state_range_raw.astype(np.float32),
            "safe_state_range_raw": safe_state_range_raw.astype(np.float32),
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
            "num_input_features": len(self.LIMO_INPUT_COLS),
            "num_output_features": len(self.LIMO_TARGET_COLS),
            "X_all_scaled_csv_path": X_all_scaled_csv_path,
        }


class LIMO_Dataset(Dataset): 
    def __init__(self, X,y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.float32)  

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        inputs = self.X[idx].squeeze(0)
        if self.y is None:
            return inputs
        return inputs, self.y[idx].squeeze(0)


class RNN_model(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        fc_sizes: list,
        rnn_dropout_prob: float,
        fc_dropout_prob: float,
        output_size: int,
    ):

        super().__init__()
    
        # Vanilla RNN with ReLu
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            nonlinearity='relu',
            batch_first=True,
            dropout=0
        )

        self.rnn_dropout_prob = rnn_dropout_prob

        # apply dropout to the last hidden layer
        self.last_hidden_layer_dropout = nn.Dropout(self.rnn_dropout_prob)

        # Fully connected layers with ReLu
        self.fc_dropout_prob = fc_dropout_prob
        self.fc_layers = []
        prev_layer = hidden_size
        self.fc_sizes = fc_sizes
        for s in self.fc_sizes:
            self.fc_layers.append(nn.Linear(prev_layer, s))
            self.fc_layers.append(nn.ReLU())   
            self.fc_layers.append(nn.Dropout(self.fc_dropout_prob))  # FC dropout layer
            prev_layer = s

        output_layer = nn.Linear(prev_layer, output_size)  # output next state

        self.fc_layers.append(output_layer)
        self.fc = nn.Sequential(*self.fc_layers)

        self.init_weights()
    
    def init_weights(self):
        # RNN initialization
        for name, param in self.rnn.named_parameters():
            print(f"RNN_Name: {name} ---- Shape: {param.shape}")
            if "weight_ih" in name:
                # nn.init.xavier_uniform_(param) # weight init from xavier uniform(-a,a)
                nn.init.kaiming_uniform_(param) 
                # print(f"weight_ih_init:{param}")
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        # FC layers initialization
        for i, layer in enumerate(self.fc):
            if isinstance(layer, nn.Linear):
                if i == len(self.fc) - 1:
                    nn.init.xavier_uniform_(layer.weight)
                else:
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        # print(f"input_shape in forward:{x.shape}")
        rnn_out, h = self.rnn(x) # output: tensor of shape (L,D∗Hout)(L,D∗Hout​)  h: tensor of shape (D*num_layers,Hout​), D = 2 if bidirectional=True otherwise 1
        # print(f"rnn_output:{rnn_out.shape}, hidden_state:{h}")
        last_hidden_state = h[-1]       
        # print(f"last_hidden_state_shape:{last_hidden_state.shape}")
        h_dropped = self.last_hidden_layer_dropout(last_hidden_state)
        # print(f"hidden_state_after_dropout_shape:{h_dropped.shape}")
        output = self.fc(h_dropped)
        # print(f"output_shape:{output.shape}")
        return output


class RNN_trainer(object):
    def __init__(self, model:nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, 
                 lr: float,weight_decay: float, epochs: int,model_dir:str, selected_vars: list, scaler, patience: int):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        # self.device = device
        self.epochs = epochs
        # self.model_save_path = model_save_path
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.model_dir = model_dir
        self.selected_vars = selected_vars
        self.scaler = scaler

        # Split states into pose vs velo channels for mixed-objective learning.
        self.pose_var_names = {"x", "y", "yaw"}
        self.velo_var_names = {"v", "w"}
        self.pose_indices = [i for i, v in enumerate(self.selected_vars) if v in self.pose_var_names]
        self.velo_indices = [i for i, v in enumerate(self.selected_vars) if v in self.velo_var_names]
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr,weight_decay=self.weight_decay)
        # self.t_total = len(train_loader) * epochs
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min',factor=0.2,patience=4)
        self.pose_loss_fn = nn.SmoothL1Loss(beta=0.5)
        self.velo_loss_fn = nn.MSELoss()
        self.pose_loss_weight = 1.2
        self.velo_loss_weight = 1.5
        self.y_scaler = None
        self.state_range_raw = None
        if isinstance(self.scaler, dict):
            self.y_scaler = self.scaler.get("y_scaler")
            self.state_range_raw = self.scaler.get("state_range_raw")
        else:
            self.y_scaler = self.scaler

        # if not using state range raw data then use state range from scaler
        if self.state_range_raw is None and self.y_scaler is not None and hasattr(self.y_scaler, "data_range_"):
            self.state_range_raw = np.asarray(self.y_scaler.data_range_, dtype=np.float32)

        # if not using state range raw data and y scaler is none then use 1 as range, so that in compute error the error does not change
        if self.state_range_raw is None:
            self.state_range_raw = np.ones(len(self.selected_vars), dtype=np.float32)

        self.safe_state_range_raw = np.where(
            np.abs(self.state_range_raw) < 1e-8, 1.0, self.state_range_raw
        ).astype(np.float32)
   
    def normalized_loss(self, pred_state: torch.Tensor, target_state: torch.Tensor):
        error_scaled = pred_state - target_state
        error_raw = error_scaled

        # convert scaled error to raw error
        if self.y_scaler is not None and hasattr(self.y_scaler, "scale_"):
            scale_factor = np.asarray(self.y_scaler.scale_, dtype=np.float32)
            safe_scale_factor = np.where(np.abs(scale_factor) < 1e-8, 1.0, scale_factor)
            safe_scale = torch.as_tensor(
                safe_scale_factor, dtype=error_scaled.dtype, device=error_scaled.device
            )
            if isinstance(self.y_scaler, MinMaxScaler):
                # MinMax: diff_scaled = diff_raw * scale => diff_raw = diff_scaled / scale
                error_raw = error_scaled / safe_scale
            elif isinstance(self.y_scaler, StandardScaler):
                # Standard: diff_scaled = diff_raw / scale => diff_raw = diff_scaled * scale
                error_raw = error_scaled * safe_scale

        safe_range = torch.as_tensor(
            self.safe_state_range_raw, dtype=error_raw.dtype, device=error_raw.device
        )

        normalized_error = error_raw / safe_range

        # Use different metrics for pose states(x,y,yaw) and velocity states (v and w).
        total_loss = torch.tensor(0.0, dtype=normalized_error.dtype, device=normalized_error.device)
        active_weight = 0.0

        if len(self.pose_indices) > 0:
            pose_error = normalized_error[:, self.pose_indices]
            pose_loss = self.pose_loss_fn(pose_error, torch.zeros_like(pose_error))
            total_loss = total_loss + self.pose_loss_weight * pose_loss
            active_weight += self.pose_loss_weight

        if len(self.velo_indices) > 0:
            velo_error = normalized_error[:, self.velo_indices]
            velo_loss = self.velo_loss_fn(velo_error, torch.zeros_like(velo_error))
            total_loss = total_loss + self.velo_loss_weight * velo_loss
            active_weight += self.velo_loss_weight

        if active_weight <= 0.0:
            return self.pose_loss_fn(normalized_error, torch.zeros_like(normalized_error))

        return total_loss / active_weight
    
    def train(self):
        print("======================== Begin Training ========================")
        self.model.train()
        losses = []
        all_preds=[]
        all_targets =[]
        for idx, (x_batch,y_batch) in enumerate(self.train_loader):
            # print(type(x_batch))
            # print(len(x_batch))
            # print("======================",x_batch[0],type(x_batch[0]), x_batch[0].shape)
            # print("=============++++++++===========",y_batch[0],type(y_batch[0]), y_batch[0].shape)

            # x_batch = x_batch.to(self.device)
            # y_batch = y_batch.to(self.device)
            pred_state = self.model(x_batch)
            # print("pred_rul type:",type(pred_rul))
            loss = self.normalized_loss(pred_state, y_batch)
            # print(f"pred_rul: {pred_rul}")
            # print(f"true_rul: {y_batch}")

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0)

            self.optimizer.step()

            all_preds.append(pred_state.detach().cpu())
            all_targets.append(y_batch.detach().cpu())
            

            losses.append(loss.item())
            # print("loss:",loss.item())
                                
        avg_loss = np.mean(losses)
        # self.scheduler.step(avg_loss)
        # print("Learning rate:", self.optimizer.param_groups[0]['lr'])

        return avg_loss
    
    def validate(self):
        print("======================== Begin Validation ========================")
        self.model.eval()
        losses = []
        all_preds=[]
        all_targets =[]
        with torch.no_grad():
            for idx, (x_batch,y_batch) in enumerate(self.val_loader):
                # x_batch = x_batch.to(self.device)
                # y_batch = y_batch.to(self.device)
                pred_state = self.model(x_batch)
                loss = self.normalized_loss(pred_state, y_batch)
                losses.append(loss.item())
                all_preds.append(pred_state.detach().cpu())
                all_targets.append(y_batch.detach().cpu())
                                
        avg_loss = np.mean(losses)
        return avg_loss


    def save_model(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            print("Created model directory:", self.model_dir)
    
        current_val_loss = float("inf")
        best_model_path = self.model_dir + "/best_RNN_next_state_model.pth"
        wait = 0
        train_losses = []
        val_losses = []
        for e in range(self.epochs):
            print(f"======== Epoch {e+1}/{self.epochs} ========")
            train_loss = self.train()
            val_loss = self.validate()
            self.scheduler.step(val_loss)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print(f"Epoch {e+1} | "
                        f"Train {train_loss:.3f} | "
                        f"Val {val_loss:.3f} | "
                        f"LR {self.optimizer.param_groups[0]['lr']}")
            # save best model
            if val_loss < current_val_loss:
                current_val_loss = val_loss
                wait = 0
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "input_size": self.model.rnn.input_size,
                    "hidden_size": self.model.rnn.hidden_size,
                    "output_size": self.model.fc[-1].out_features,
                    "fc_sizes": self.model.fc_sizes,  
                    "rnn_dropout_prob": self.model.rnn_dropout_prob,
                    "fc_dropout_prob": self.model.fc_dropout_prob,
                    "selected_vars": self.selected_vars,
                    "loss_config": {
                        "pose_vars": sorted(list(self.pose_var_names)),
                        "velo_vars": sorted(list(self.velo_var_names)),
                        "pose_loss": "SmoothL1(beta=0.5)",
                        "velo_loss": "MSE",
                        "pose_weight": self.pose_loss_weight,
                        "velo_weight": self.velo_loss_weight,
                    },
                    "scaler": self.scaler,
                }, best_model_path)

                print("Saved best model to:", best_model_path)
            else:
                wait += 1
                if wait >= self.patience:
                    print(f"Early stopping at epoch {e+1}")
                    break

        return best_model_path,train_losses,val_losses
        

def predict(model, X_test, batch_size=32):
    
    model.eval()
    ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for (x_batch,) in dl:
            p = model(x_batch).detach().cpu().numpy()
            preds.append(p)

    return np.concatenate(preds, axis=0)


def plot_corelation_heatmap(df_train):
    plt.figure(figsize=(12,10))
    threshold = 0.5
    corr = df_train.corr()
    mask = corr.where((abs(corr) >= threshold)).isna()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, mask=mask,linewidths=0.2, 
            linecolor='lightgrey').set_facecolor('white')
    plt.title('Feature Correlation Heatmap', fontsize=16)
    plt.show()

def plot_loss_curve(train_losses, val_losses,save_path="./"):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid()
    plt.savefig(save_path + '/loss_curve_RNN.png')
    plt.show()
    # plt.close()
    

def plot_prediction_vs_origin_over_time(
    pred_raw: np.ndarray,
    y_raw: np.ndarray,
    state_cols: list,
    save_path: str = "./",
    max_points: Optional[int] = None,
):
    if pred_raw.shape != y_raw.shape:
        raise RuntimeError(
            f"Shape mismatch for plotting: pred_raw {pred_raw.shape} vs y_raw {y_raw.shape}"
        )
    if pred_raw.size == 0:
        print("No prediction points available for plotting.")
        return

    n_plot = min(len(pred_raw), max_points) if max_points is not None else len(pred_raw)
    pred_plot = pred_raw[:n_plot]
    y_plot = y_raw[:n_plot]
    t = np.arange(n_plot)

    n_states = pred_plot.shape[1]
    fig, axes = plt.subplots(n_states, 1, figsize=(12, 2.6 * n_states), sharex=True)
    if n_states == 1:
        axes = [axes]

    for i, c in enumerate(state_cols):
        axes[i].plot(t, y_plot[:, i], label="origin", linewidth=1.6)
        axes[i].plot(t, pred_plot[:, i], label="pred", linewidth=1.2, linestyle="--")
        axes[i].set_ylabel(c)
        axes[i].grid(alpha=0.3)
        if i == 0:
            axes[i].legend(loc="upper right")
    axes[-1].set_xlabel("Validation step index")
    fig.suptitle("Predicted vs Origin State Over Time")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out1 = os.path.join(save_path, "pred_vs_origin_over_time.png")
    fig.savefig(out1, dpi=200)
    plt.show()

    err = pred_plot - y_plot
    fig2, axes2 = plt.subplots(n_states, 1, figsize=(12, 2.6 * n_states), sharex=True)
    if n_states == 1:
        axes2 = [axes2]
    for i, c in enumerate(state_cols):
        axes2[i].plot(t, err[:, i], color="tab:red", linewidth=1.2)
        axes2[i].axhline(0.0, color="black", linewidth=0.8, linestyle=":")
        axes2[i].set_ylabel(f"{c} err")
        axes2[i].grid(alpha=0.3)
    axes2[-1].set_xlabel("Validation step index")
    fig2.suptitle("Prediction Error Over Time (pred - origin)")
    fig2.tight_layout(rect=[0, 0, 1, 0.98])
    out2 = os.path.join(save_path, "pred_error_over_time.png")
    fig2.savefig(out2, dpi=200)
    plt.show()

    print(f"Saved plot: {out1}")
    print(f"Saved plot: {out2}")


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)




##### Begin Evaluation ################
def inverse_transform_states(state_scaled: np.ndarray, y_scaler):
    if y_scaler is None:
        return state_scaled.astype(np.float32)
    if not hasattr(y_scaler, "inverse_transform"):
        raise RuntimeError("Provided y_scaler has no inverse_transform for state reconstruction.")
    return y_scaler.inverse_transform(state_scaled).astype(np.float32)

def normalize_error_by_state_range(error_raw: np.ndarray, state_range_raw: np.ndarray):
    if error_raw.ndim != 2:
        raise RuntimeError(f"Expected error_raw shape (N, F), got {error_raw.shape}")
    state_range_raw = np.asarray(state_range_raw, dtype=np.float32)
    if state_range_raw.ndim == 0:
        state_range_raw = np.full((error_raw.shape[1],), float(state_range_raw), dtype=np.float32)
    if state_range_raw.ndim != 1 or state_range_raw.shape[0] != error_raw.shape[1]:
        raise RuntimeError(
            f"Expected state_range_raw shape ({error_raw.shape[1]},), got {state_range_raw.shape}"
        )
    safe_range = np.where(np.abs(state_range_raw) < 1e-8, 1.0, state_range_raw).astype(np.float32)
    return error_raw / safe_range, safe_range


def evaluate_next_state_with_normalized_error(
    model: nn.Module,
    X_val: np.ndarray,
    y_val_raw: np.ndarray,
    state_range_raw: Optional[np.ndarray],
    y_scaler,
    state_cols: list,
    plot_dir: str = "./",
):
    pred_scaled = predict(model, X_val)
    print("X_val:", X_val.shape)   # (N, T, F_in)
    print("pred_scaled:", pred_scaled.shape)     # (N, F_out)
    pred_raw = inverse_transform_states(pred_scaled, y_scaler)

    if pred_raw.shape != y_val_raw.shape:
        raise RuntimeError(
            f"Shape mismatch: pred {pred_raw.shape} vs target {y_val_raw.shape}"
        )

    num_pred=50
    print(f"\nFirst {num_pred} predicted states:")
    pred_df = pd.DataFrame(pred_raw[:num_pred], columns=state_cols)
    print(pred_df)
    print(f"\nFirst {num_pred} origin states:")
    target_df = pd.DataFrame(y_val_raw[:num_pred], columns=state_cols)
    print(target_df)

    plot_prediction_vs_origin_over_time(
        pred_raw=pred_raw,
        y_raw=y_val_raw,
        state_cols=state_cols,
        save_path=plot_dir,
    )

    raw_error = pred_raw - y_val_raw
    raw_rmse_each = np.sqrt(np.mean(raw_error ** 2, axis=0))
    print("\nValidation raw RMSE per state:")
    for c, r in zip(state_cols, raw_rmse_each):
        print(f"  {c}: {r:.6f}")
    print(f"Validation raw RMSE mean: {raw_rmse_each.mean():.6f}")

    if state_range_raw is None:
        print(
            "Warning: state_range_raw not provided. Falling back to target range on validation split."
        )
        state_range_raw = (
            np.max(y_val_raw, axis=0) - np.min(y_val_raw, axis=0)
        ).astype(np.float32)
    normalized_error, safe_state_range = normalize_error_by_state_range(
        raw_error, state_range_raw.astype(np.float32)
    )
    abs_normalized_error = np.abs(normalized_error)
    nmae_each = np.mean(abs_normalized_error, axis=0)
    nrmse_each = np.sqrt(np.mean(normalized_error ** 2, axis=0))

    print("\nValidation range-normalized MAE per state (|pred-true| / range):")
    for c, n in zip(state_cols, nmae_each):
        print(f"  {c}: {n:.6f} ({100.0 * n:.2f}%)")
    print(
        f"Validation range-normalized MAE mean: {nmae_each.mean():.6f} "
        f"({100.0 * nmae_each.mean():.2f}%)"
    )

    print("\nValidation range-normalized RMSE per state (sqrt(mean((pred-true)^2)) / range):")
    for c, n in zip(state_cols, nrmse_each):
        print(f"  {c}: {n:.6f} ({100.0 * n:.2f}%)")
    print(
        f"Validation range-normalized RMSE mean: {nrmse_each.mean():.6f} "
        f"({100.0 * nrmse_each.mean():.2f}%)"
    )

    pose_vars = {"x", "y", "yaw"}
    velo_vars = {"v", "w"}
    pose_idx = [i for i, c in enumerate(state_cols) if c in pose_vars]
    velo_idx = [i for i, c in enumerate(state_cols) if c in velo_vars]

    if len(pose_idx) > 0:
        pose_nrmse_mean = float(np.mean(nrmse_each[pose_idx]))
        print(
            f"Pose normalized RMSE mean (x,y,yaw): {pose_nrmse_mean:.6f} "
            f"({100.0 * pose_nrmse_mean:.2f}%)"
        )
    if len(velo_idx) > 0:
        velo_nrmse_mean = float(np.mean(nrmse_each[velo_idx]))
        print(
            f"Control normalized RMSE mean (v,w): {velo_nrmse_mean:.6f} "
            f"({100.0 * velo_nrmse_mean:.2f}%)"
        )

    print("\nState ranges used for normalization:")
    for c, r in zip(state_cols, safe_state_range):
        print(f"  {c}: {r:.6f}")


if __name__ == "__main__":

    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)
    set_seed(25)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_csv = os.path.join(
        script_dir, "util/data/LIMO_trajectories/limo_traj/rosbag1_4f.csv"
    )
    model_dir = os.path.join(script_dir, "util/data/LIMO_trajectories/saved_models")
    processed_dir = os.path.join(script_dir, "util/data/LIMO_trajectories/limo_processed")

    win_size = 20
    val_ratio = 0.2
    batch_size = 32
    # Use command channels as input context and predict next robot state.
    LIMO_INPUT_COLS = ["x", "y", "yaw", "v", "w", "v_cmd", "w_cmd"]
    LIMO_TARGET_COLS = ["x", "y", "yaw", "v", "w"]
    scaler = MinMaxScaler()
    trajectory_processor = RNN_trajectory(
        traj_path=data_csv,
        scaler = scaler,
        val_ratio=val_ratio,
        win_size=win_size,
        LIMO_INPUT_COLS=LIMO_INPUT_COLS,
        LIMO_TARGET_COLS=LIMO_TARGET_COLS,
    )
    data = trajectory_processor.process_limo_next_state_dataset(
        processed_traj_path=processed_dir
    )
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    input_cols = data["input_cols"]
    target_cols = data["target_cols"]
    num_input_features = data["num_input_features"]
    num_output_features = data["num_output_features"]

    print("Loaded trajectory file:", data_csv)
    print("Raw rows:", len(data["df_raw"]), "| Clean rows:", len(data["df_new"]))
    print("Input columns:", input_cols)
    print("Target columns:", target_cols)
    print("Training target: full next state S(t+1)")
    print(
        "Training loss: mixed normalized error "
        "(pose x,y,yaw -> SmoothL1, velo v,w -> MSE, velo weight=1.5)"
    )
    print("Window size:", win_size)
    print("Train windows:", X_train.shape, "Targets:", y_train.shape)
    print("Val windows:", X_val.shape, "Targets:", y_val.shape)
    if data["X_all_scaled_csv_path"] is not None:
        print("Saved scaled full input CSV:", data["X_all_scaled_csv_path"])

    train_dataset = LIMO_Dataset(X_train, y_train)
    val_dataset = LIMO_Dataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = RNN_model(
        input_size=num_input_features,
        hidden_size=32,
        fc_sizes=[32,16],
        rnn_dropout_prob=0.0,
        fc_dropout_prob=0.0,
        output_size=num_output_features,
    )

    scaler_bundle = {
        "x_scaler": data["x_scaler"],
        "y_scaler": data["y_scaler"],
        "state_range_raw": data["safe_state_range_raw"],
    }
    # scaler_bundle =None

    trainer = RNN_trainer(
        model,
        train_loader,
        val_loader,
        lr=1e-3,
        weight_decay=1e-4,
        epochs=100,
        model_dir=model_dir,
        selected_vars=target_cols,
        scaler=scaler_bundle,
        patience=6,
    )
    save_model_path, train_losses, val_losses = trainer.save_model()
    plot_loss_curve(train_losses, val_losses, model_dir)

    checkpoint = torch.load(
        save_model_path, map_location=torch.device("cpu"), weights_only=False
    )
    model = RNN_model(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        fc_sizes=checkpoint["fc_sizes"],
        rnn_dropout_prob=checkpoint["rnn_dropout_prob"],
        fc_dropout_prob=checkpoint["fc_dropout_prob"],
        output_size=checkpoint["output_size"],
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    evaluate_next_state_with_normalized_error(
        model=model,
        X_val=X_val,
        y_val_raw=data["y_val_raw"],
        state_range_raw=data["state_range_raw"],
        y_scaler=data["y_scaler"],
        state_cols=target_cols,
        plot_dir=model_dir,
    )
