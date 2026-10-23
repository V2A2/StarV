# export_ros2bag_limo_traj.py
# Output: one CSV per bag in ./traj_csv with columns:
# t, x, y, yaw, v, a, w, v_cmd, w_cmd

import math
from pathlib import Path
import pandas as pd
import numpy as np

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def compute_yaw(qx, qy, qz, qw): 
    # yaw about z-axis
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def read_bag(bag_dir: str):
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=bag_dir, storage_id="sqlite3"),
        ConverterOptions("cdr", "cdr"),
    )

    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}

    need = {"/odom", "/cmd_vel"}
    for t in need:
        if t not in topic_types:
            raise RuntimeError(f"Missing topic {t}. Found: {list(topic_types.keys())}")

    Odom = get_message(topic_types["/odom"])
    Twist = get_message(topic_types["/cmd_vel"])

    odom_rows = []
    cmd_rows = []
    t0_ns = None

    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if t0_ns is None:
            t0_ns = t_ns
        t = (t_ns - t0_ns) * 1e-9  # seconds from bag start

        if topic == "/odom":
            msg = deserialize_message(data, Odom)

            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            q = msg.pose.pose.orientation
            yaw = compute_yaw(q.x, q.y, q.z, q.w)

            vx = msg.twist.twist.linear.x
            vy = msg.twist.twist.linear.y
            v = np.sqrt(vx**2 + vy**2)  # speed in the plane

            w = msg.twist.twist.angular.z  # yaw rate 

            odom_rows.append({"t": t, "x": x, "y": y, "yaw": yaw, "v": v, "w": w})

        elif topic == "/cmd_vel":
            msg = deserialize_message(data, Twist)
            cmd_rows.append({"t": t, "v_cmd": msg.linear.x, "w_cmd": msg.angular.z})

    odom = pd.DataFrame(odom_rows).sort_values("t")
    cmd = pd.DataFrame(cmd_rows).sort_values("t")
    return odom, cmd


def add_accel(df: pd.DataFrame):
    df = df.sort_values("t").copy()
    dt = df["t"].diff()
    dv = df["v"].diff()
    df["a"] = dv / dt
    df.loc[(dt <= 1e-6) | (~dt.notna()), "a"] = float("nan")
    return df


def merge_cmd(odom: pd.DataFrame, cmd: pd.DataFrame):
    if odom.empty:
        return odom
    if cmd.empty:
        odom["v_cmd"] = float("nan")
        odom["w_cmd"] = float("nan")
        return odom

    return pd.merge_asof(
        odom.sort_values("t"),
        cmd.sort_values("t"),
        on="t",
        direction="nearest",
        tolerance=0.05, 
    )


def downsample_to_step_size(df: pd.DataFrame, step=0.1): # hz used for trajectory gneraton in simulation
    df = df.sort_values("t").copy()
    if df.empty:
        return df

    # Ensure exported trajectory time starts at 0.
    df["t"] = df["t"] - df["t"].iloc[0]
    df["t"] = (df["t"] / step).round() * step

    out = df.groupby("t", as_index=False).last().sort_values("t")
    out["t"] = out["t"] - out["t"].iloc[0]
    out["t"] = out["t"].round(6)
    return out


def round_state(df: pd.DataFrame, decimals=8):
    output_cols = ["x", "y", "yaw", "v", "w", "a", "v_cmd", "w_cmd"]
    for col in output_cols:
        if col in df.columns:
            df[col] = df[col].round(decimals)
    return df


def find_bag_dirs(bag_root: Path):
    if not bag_root.exists():
        raise RuntimeError(f"Bag root does not exist: {bag_root}")

    bag_dirs = sorted({p.parent for p in bag_root.rglob("metadata.yaml")})
    if not bag_dirs:
        raise RuntimeError(f"No rosbag folders found under: {bag_root}")
    return bag_dirs


def extract_single_bag(bag_dir: Path, out_csv: Path, step=0.1):
    odom, cmd = read_bag(str(bag_dir))
    # odom = add_accel(odom)  # no need to add acceleration, a can be derived from v.
    merged = merge_cmd(odom, cmd)
    out = downsample_to_step_size(merged, step=step)
    out = round_state(out, decimals=8)
    out.to_csv(out_csv, index=False)
    return out


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    print(f"Script directory: {script_dir}")

    bag_root = script_dir / "rnn_traj_rosbags"
    bag_dirs = find_bag_dirs(bag_root)
    print(f"Found {len(bag_dirs)} rosbag folders under {bag_root}")

    save_traj_dir = script_dir / "traj_csv"
    save_traj_dir.mkdir(parents=True, exist_ok=True)
    print(f"Trajectory output directory: {save_traj_dir}")

    for bag_dir in bag_dirs:
        out_csv = save_traj_dir / f"{bag_dir.name}_4f_test.csv"
        out = extract_single_bag(bag_dir, out_csv, step=0.1)
        print(f"{bag_dir.name}: Extracted {len(out)} samples to {out_csv}")
