"""
RNN training for LIMO simulation
Qing Liu, 3/6/2025
"""
import random
from scipy.io import loadmat
import os
import json
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
            for name, param in layer.named_parameters():
                if isinstance(layer, nn.Linear):
                    # print(f"FC_layer {i}: {layer} ---- Shape: {param.shape}")
                    # if isinstance(param, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight,nonlinearity='relu')
                    # print(f"FC_layer {i} weight initialization:{layer.weight}")
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
                 lr: float,weight_decay: float, epochs: int,model_dir:str, selected_vars: list, scaler: StandardScaler, patience: int):
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
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr,weight_decay=self.weight_decay)
        # self.t_total = len(train_loader) * epochs
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min',factor=0.2,patience=4)
        self.loss_fn = nn.SmoothL1Loss(beta=10.0)
        # self.loss_fn = nn.MSELoss()
        
    
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
            pred_rul = self.model(x_batch)
            # print("pred_rul type:",type(pred_rul))
            loss = self.loss_fn(pred_rul,y_batch)
            # print(f"pred_rul: {pred_rul}")
            # print(f"true_rul: {y_batch}")

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0)

            self.optimizer.step()

            all_preds.append(pred_rul.detach().cpu())
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
                pred_rul = self.model(x_batch)
                loss = self.loss_fn(pred_rul,y_batch)
                losses.append(loss.item())
                all_preds.append(pred_rul.detach().cpu())
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
                    "scaler": self.scaler,
                }, best_model_path)

                print("Saved best model to:", best_model_path)
            else:
                wait += 1
                if wait >= self.patience:
                    print(f"Early stopping at epoch {e+1}")
                    break

        return best_model_path,train_losses,val_losses
        

def predict(model, X_test, batch_size=256):
    model.eval()
    ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for (x_batch,) in dl:
            p = model(x_batch).detach().cpu().numpy().reshape(-1)
            preds.append(p)

    return np.concatenate(preds)

def plot_egine_cycles(df_train,index_names):

        max_time_cycles=df_train[index_names].groupby('unit_number').max()
        plt.figure(figsize=(20,50))
        ax=max_time_cycles['time_cycles'].plot(kind='barh',width=0.8, stacked=True,align='center')
        plt.title('Turbofan Engines LifeTime',fontweight='bold',size=30)
        plt.xlabel('Time cycle',fontweight='bold',size=20)
        plt.xticks(size=15)
        plt.ylabel('unit',fontweight='bold',size=20)
        plt.yticks(size=15)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

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
    

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_next_state_windows(sequence_data: np.ndarray, window_size: int):
    if sequence_data.shape[0] <= window_size:
        raise RuntimeError(
            f"Not enough steps ({sequence_data.shape[0]}) for window size {window_size}"
        )

    X = []
    y = []
    for start in range(sequence_data.shape[0] - window_size):
        end = start + window_size
        X.append(sequence_data[start:end, :])
        y.append(sequence_data[end, :])  # next step (T+1)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


LIMO_STATE_COLS = ["x", "y", "yaw", "v", "w", "v_cmd", "w_cmd"]


def clean_limo_trajectory_df(df: pd.DataFrame, state_cols: list):
    clean_df = df.copy()

    if "t" in clean_df.columns:
        clean_df = clean_df.sort_values("t").drop_duplicates(subset="t", keep="first")

    for col in state_cols:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    before = len(clean_df)
    clean_df = clean_df.dropna(subset=state_cols).reset_index(drop=True)
    removed = before - len(clean_df)
    if removed > 0:
        print(f"Removed {removed} rows with NaN/invalid values in state columns.")

    return clean_df


def split_windows_by_time(X_all: np.ndarray, y_all: np.ndarray, val_ratio: float):
    split_idx = int((1.0 - val_ratio) * len(X_all))
    if split_idx <= 0 or split_idx >= len(X_all):
        raise RuntimeError(
            f"Invalid split index {split_idx} for {len(X_all)} windows. "
            f"Adjust val_ratio={val_ratio}."
        )

    X_train_raw = X_all[:split_idx]
    y_train_raw = y_all[:split_idx]
    X_val_raw = X_all[split_idx:]
    y_val_raw = y_all[split_idx:]
    return X_train_raw, y_train_raw, X_val_raw, y_val_raw, split_idx


def process_limo_next_state_dataset(
    csv_path: str,
    window_size: int,
    val_ratio: float = 0.2,
    processed_dir: Optional[str] = None,
):
    csv_path_obj = Path(csv_path)
    df_raw = pd.read_csv(csv_path_obj)

    missing = [c for c in LIMO_STATE_COLS if c not in df_raw.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in {csv_path}: {missing}")

    df_clean = clean_limo_trajectory_df(df_raw, LIMO_STATE_COLS)
    sequence_data = df_clean[LIMO_STATE_COLS].to_numpy(dtype=np.float32)
    X_all, y_all = build_next_state_windows(sequence_data, window_size)
    X_train_raw, y_train_raw, X_val_raw, y_val_raw, split_idx = split_windows_by_time(
        X_all, y_all, val_ratio
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_2d = X_train_raw.reshape(-1, X_train_raw.shape[-1])
    X_val_2d = X_val_raw.reshape(-1, X_val_raw.shape[-1])

    X_train_scaled = x_scaler.fit_transform(X_train_2d).reshape(X_train_raw.shape)
    X_val_scaled = x_scaler.transform(X_val_2d).reshape(X_val_raw.shape)
    y_train_scaled = y_scaler.fit_transform(y_train_raw)
    y_val_scaled = y_scaler.transform(y_val_raw)

    dt_median = None
    dt_min = None
    dt_max = None
    if "t" in df_clean.columns and len(df_clean) > 1:
        dt = np.diff(df_clean["t"].to_numpy(dtype=np.float64))
        if dt.size > 0:
            dt_median = float(np.median(dt))
            dt_min = float(np.min(dt))
            dt_max = float(np.max(dt))

    processed_npz_path = None
    processed_meta_path = None
    if processed_dir is not None:
        processed_dir_path = Path(processed_dir)
        processed_dir_path.mkdir(parents=True, exist_ok=True)

        dataset_tag = f"{csv_path_obj.stem}_win{window_size}_next_state"
        processed_npz_path = processed_dir_path / f"{dataset_tag}.npz"
        processed_meta_path = processed_dir_path / f"{dataset_tag}_meta.json"

        np.savez_compressed(
            processed_npz_path,
            X_train=X_train_scaled.astype(np.float32),
            y_train=y_train_scaled.astype(np.float32),
            X_val=X_val_scaled.astype(np.float32),
            y_val=y_val_scaled.astype(np.float32),
            y_val_raw=y_val_raw.astype(np.float32),
        )

        meta = {
            "csv_path": str(csv_path_obj),
            "state_columns": LIMO_STATE_COLS,
            "window_size": int(window_size),
            "val_ratio": float(val_ratio),
            "num_rows_raw": int(len(df_raw)),
            "num_rows_clean": int(len(df_clean)),
            "num_windows_total": int(len(X_all)),
            "num_windows_train": int(len(X_train_raw)),
            "num_windows_val": int(len(X_val_raw)),
            "split_idx": int(split_idx),
            "dt_median": dt_median,
            "dt_min": dt_min,
            "dt_max": dt_max,
        }
        with open(processed_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    return {
        "df_raw": df_raw,
        "df_clean": df_clean,
        "state_cols": LIMO_STATE_COLS,
        "X_train": X_train_scaled.astype(np.float32),
        "y_train": y_train_scaled.astype(np.float32),
        "X_val": X_val_scaled.astype(np.float32),
        "y_val": y_val_scaled.astype(np.float32),
        "y_val_raw": y_val_raw.astype(np.float32),
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "num_features": len(LIMO_STATE_COLS),
        "processed_npz_path": processed_npz_path,
        "processed_meta_path": processed_meta_path,
        "dt_median": dt_median,
        "dt_min": dt_min,
        "dt_max": dt_max,
    }


def evaluate_next_state_rmse(model: nn.Module, X_val: np.ndarray, y_val_raw: np.ndarray, y_scaler: StandardScaler, state_cols: list):
    pred_scaled = predict(model, X_val)
    pred_raw = y_scaler.inverse_transform(pred_scaled)
    rmse_each = np.sqrt(np.mean((pred_raw - y_val_raw) ** 2, axis=0))

    print("\nValidation RMSE per state:")
    for c, r in zip(state_cols, rmse_each):
        print(f"  {c}: {r:.6f}")
    print(f"Validation RMSE mean: {rmse_each.mean():.6f}")


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
    batch_size = 32

    data = process_limo_next_state_dataset(
        csv_path=data_csv,
        window_size=win_size,
        val_ratio=0.2,
        processed_dir=processed_dir,
    )
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    state_cols = data["state_cols"]
    num_features = data["num_features"]

    print("Loaded trajectory file:", data_csv)
    print("Raw rows:", len(data["df_raw"]), "| Clean rows:", len(data["df_clean"]))
    if data["dt_median"] is not None:
        print(
            f"dt stats -> median: {data['dt_median']:.6f}, "
            f"min: {data['dt_min']:.6f}, max: {data['dt_max']:.6f}"
        )
    print("State columns:", state_cols)
    print("Window size:", win_size)
    print("Train windows:", X_train.shape, "Targets:", y_train.shape)
    print("Val windows:", X_val.shape, "Targets:", y_val.shape)
    if data["processed_npz_path"] is not None:
        print("Saved processed dataset:", data["processed_npz_path"])
    if data["processed_meta_path"] is not None:
        print("Saved processing metadata:", data["processed_meta_path"])

    train_dataset = LIMO_Dataset(X_train, y_train)
    val_dataset = LIMO_Dataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = RNN_model(
        input_size=num_features,
        hidden_size=32,
        fc_sizes=[64, 32, 16],
        rnn_dropout_prob=0.2,
        fc_dropout_prob=0.2,
        output_size=num_features,
    )

    scaler_bundle = {"x_scaler": data["x_scaler"], "y_scaler": data["y_scaler"]}

    trainer = RNN_trainer(
        model,
        train_loader,
        val_loader,
        lr=1e-3,
        weight_decay=1e-4,
        epochs=30,
        model_dir=model_dir,
        selected_vars=state_cols,
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

    evaluate_next_state_rmse(
        model=model,
        X_val=X_val,
        y_val_raw=data["y_val_raw"],
        y_scaler=data["y_scaler"],
        state_cols=state_cols,
    )
