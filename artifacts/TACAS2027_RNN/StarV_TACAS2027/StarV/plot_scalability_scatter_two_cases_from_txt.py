#!/usr/bin/env python3
"""
Plot scalability scatter from two verification logs (LIMO and CMAPSS):
  x-axis: total branches after reachability
  y-axis: verification time t_v = t_r + t_c

Style:
  - log-log scatter
  - color by case study
  - marker by T
  - dashed log-log trendline per case

Outputs:
  - parsed CSV
  - PNG figure
"""

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


RUN_RE = re.compile(
    r"^=+\s*(LIMO|CMAPSS|CMPASS) verification:\s*T=(\d+),\s*p_f=([0-9eE+\-.]+)\s*=+"
)
REACH_RE = re.compile(r"^Reachability analysis time:\s*([0-9eE+\-.]+)\s*seconds")
TOTAL_BRANCH_RE = re.compile(r"^total branches after reachability:(\d+)")
SPEC_RE = re.compile(r"^=+Verify Spec\s+(\d+)\s+\((.+)\)=+")
CHECK_RE = re.compile(r"^Checking time for Spec\s+(\d+):\s*([0-9eE+\-.]+)\s*seconds")


def _normalize_case(case_name: str) -> str:
    case_name = case_name.upper()
    if case_name == "CMPASS":
        return "CMAPSS"
    return case_name


def parse_case_log(txt_path: str) -> pd.DataFrame:
    """
    Parse one log file into rows:
      [case, T, p_f, spec_idx, spec_label, total_branches, t_r, t_c, t_v]
    """
    records: List[dict] = []

    current_case: Optional[str] = None
    current_T: Optional[int] = None
    current_pf: Optional[float] = None
    current_t_r: Optional[float] = None
    current_total_branches: Optional[int] = None
    current_spec_idx: Optional[int] = None
    current_spec_label: Optional[str] = None

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()

            m_run = RUN_RE.match(line)
            if m_run:
                current_case = _normalize_case(m_run.group(1))
                current_T = int(m_run.group(2))
                current_pf = float(m_run.group(3))
                current_t_r = None
                current_total_branches = None
                current_spec_idx = None
                current_spec_label = None
                continue

            m_reach = REACH_RE.match(line)
            if m_reach:
                current_t_r = float(m_reach.group(1))
                continue

            m_branch = TOTAL_BRANCH_RE.match(line)
            if m_branch:
                current_total_branches = int(m_branch.group(1))
                continue

            m_spec = SPEC_RE.match(line)
            if m_spec:
                current_spec_idx = int(m_spec.group(1))
                current_spec_label = m_spec.group(2).strip()
                continue

            m_check = CHECK_RE.match(line)
            if m_check:
                check_idx = int(m_check.group(1))
                t_c = float(m_check.group(2))
                if (
                    current_case is None
                    or current_T is None
                    or current_pf is None
                    or current_t_r is None
                    or current_total_branches is None
                    or current_spec_idx is None
                    or current_spec_label is None
                ):
                    continue
                if check_idx != current_spec_idx:
                    continue

                records.append(
                    {
                        "case": current_case,
                        "T": current_T,
                        "p_f": current_pf,
                        "spec_idx": current_spec_idx,
                        "spec_label": current_spec_label,
                        "total_branches": current_total_branches,
                        "t_r": current_t_r,
                        "t_c": t_c,
                        "t_v": current_t_r + t_c,
                    }
                )

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(f"No parseable rows found in {txt_path}")
    return df


def _fit_loglog_trend(
    df_case: pd.DataFrame,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Fit log10(y) = a*log10(x) + b.
    Return plotted xs, ys and slope a if fit is valid.
    """
    fit_df = df_case[(df_case["total_branches"] > 0) & (df_case["t_v"] > 0)].copy()
    if fit_df.shape[0] < 2:
        return None
    if fit_df["total_branches"].nunique() < 2:
        return None

    lx = np.log10(fit_df["total_branches"].to_numpy(dtype=float))
    ly = np.log10(fit_df["t_v"].to_numpy(dtype=float))
    slope, intercept = np.polyfit(lx, ly, 1)

    x_min = float(fit_df["total_branches"].min())
    x_max = float(fit_df["total_branches"].max())
    if x_min <= 0 or x_max <= 0 or x_min == x_max:
        return None

    xs = np.logspace(np.log10(x_min), np.log10(x_max), 150)
    ys = 10 ** (intercept + slope * np.log10(xs))
    return xs, ys, float(slope)


def make_scalability_scatter(df: pd.DataFrame, out_path: str) -> None:
    """Create log-log scalability scatter with case colors, T markers, and trendlines."""
    df = df[(df["total_branches"] > 0) & (df["t_v"] > 0)].copy()
    if df.empty:
        raise RuntimeError("No positive rows for log-log plotting.")

    case_order = [c for c in ["LIMO", "CMAPSS"] if c in df["case"].unique()]
    for c in sorted(df["case"].unique()):
        if c not in case_order:
            case_order.append(c)

    colors = {
        "LIMO": "#1f77b4",
        "CMAPSS": "#d62728",
    }
    default_color = "#2ca02c"

    t_values = sorted(df["T"].unique().tolist())
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "p"]
    t_to_marker: Dict[int, str] = {
        t: marker_cycle[i % len(marker_cycle)] for i, t in enumerate(t_values)
    }

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.set_facecolor("#ececec")

    # Scatter points
    for case in case_order:
        case_df = df[df["case"] == case]
        color = colors.get(case, default_color)
        for t in t_values:
            sub = case_df[case_df["T"] == t]
            if sub.empty:
                continue
            ax.scatter(
                sub["total_branches"],
                sub["t_v"],
                s=46,
                c=color,
                marker=t_to_marker[t],
                edgecolors="black",
                linewidths=0.35,
                alpha=0.85,
            )

    # Trendlines by case
    trend_handles: List[Line2D] = []
    for case in case_order:
        case_df = df[df["case"] == case]
        fit = _fit_loglog_trend(case_df)
        if fit is None:
            continue
        xs, ys, slope = fit
        color = colors.get(case, default_color)
        line, = ax.plot(
            xs,
            ys,
            linestyle="--",
            linewidth=2.0,
            color=color,
            label=f"{case} trend (slope={slope:.2f})",
        )
        trend_handles.append(line)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total Number of Branches", fontsize=12)
    ax.set_ylabel(r"Verification Time $t_v$ (s)", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.7)

    # Legend 1: case colors + trendline slope
    case_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=colors.get(c, default_color),
            markeredgecolor="black",
            markeredgewidth=0.35,
            markersize=7,
            label=c,
        )
        for c in case_order
    ]
    if trend_handles:
        case_legend = ax.legend(
            handles=case_handles + trend_handles,
            loc="upper left",
            fontsize=9,
            frameon=True,
            title="Case / Trend",
            title_fontsize=9,
        )
    else:
        case_legend = ax.legend(
            handles=case_handles,
            loc="upper left",
            fontsize=9,
            frameon=True,
            title="Case",
            title_fontsize=9,
        )
    ax.add_artist(case_legend)

    # Legend 2: T marker map
    t_handles = [
        Line2D(
            [0],
            [0],
            marker=t_to_marker[t],
            linestyle="None",
            color="black",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=7,
            label=f"T={t}",
        )
        for t in t_values
    ]
    ax.legend(
        handles=t_handles,
        loc="lower right",
        fontsize=9,
        frameon=True,
        title="Marker by T",
        title_fontsize=9,
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create log-log scalability scatter: branches vs t_v for LIMO and CMAPSS logs."
    )
    parser.add_argument(
        "--limo-txt",
        type=str,
        default="/home/qliu1/Desktop/StarV_qing/StarV/LIMO_results_normal_noise.txt",
        help="Path to LIMO results txt file.",
    )
    parser.add_argument(
        "--cmapss-txt",
        type=str,
        default="/home/qliu1/Desktop/StarV_qing/StarV/CMAPSS_results_seed.txt",
        help="Path to CMAPSS results txt file.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/home/qliu1/Desktop/StarV_qing/StarV/results_two_cases/scalability_scatter",
        help="Output directory for figure and parsed CSV.",
    )
    parser.add_argument(
        "--case",
        type=str,
        default="both",
        choices=["both", "limo", "cmapss"],
        help="Which benchmark to include: both, limo, or cmapss.",
    )
    parser.add_argument(
        "--pf",
        type=float,
        nargs="*",
        default=None,
        help="Optional p_f filters. Example: --pf 0.0 0.05",
    )
    parser.add_argument(
        "--t-min",
        type=int,
        default=10,
        help="Minimum T to include (inclusive).",
    )
    parser.add_argument(
        "--t-max",
        type=int,
        default=20,
        help="Maximum T to include (inclusive).",
    )
    parser.add_argument(
        "--spec-idx",
        type=int,
        nargs="*",
        default=None,
        help="Optional spec indices to include (e.g., --spec-idx 0 2 4).",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    frames: List[pd.DataFrame] = []
    if args.case in ("both", "limo"):
        frames.append(parse_case_log(args.limo_txt))
    if args.case in ("both", "cmapss"):
        frames.append(parse_case_log(args.cmapss_txt))
    if not frames:
        raise RuntimeError("No input logs selected. Use --case both|limo|cmapss.")
    df = pd.concat(frames, ignore_index=True)

    # Keep intended case labels.
    df["case"] = df["case"].apply(_normalize_case)
    df = df[df["case"].isin(["LIMO", "CMAPSS"])].copy()

    # Filter by T window.
    df = df[(df["T"] >= args.t_min) & (df["T"] <= args.t_max)].copy()

    # Optional filter by p_f.
    if args.pf:
        keep = np.zeros(len(df), dtype=bool)
        for pf in args.pf:
            keep = keep | np.isclose(df["p_f"].to_numpy(dtype=float), float(pf), atol=1e-12)
        df = df[keep].copy()

    # Optional filter by spec index.
    if args.spec_idx:
        df = df[df["spec_idx"].isin(args.spec_idx)].copy()

    if df.empty:
        raise RuntimeError("No rows remain after filtering. Adjust --pf / --t-min / --t-max / --spec-idx.")

    csv_path = os.path.join(args.out_dir, "parsed_scalability_scatter_rows.csv")
    df.sort_values(["case", "T", "p_f", "spec_idx"]).to_csv(csv_path, index=False)

    fig_path = os.path.join(
        args.out_dir, f"scalability_scatter_{args.case}_branches_vs_tv_loglog.png"
    )
    make_scalability_scatter(df, fig_path)

    print(f"Saved parsed CSV: {csv_path}")
    print(f"Saved figure: {fig_path}")
    print(f"Rows plotted: {len(df)}")


if __name__ == "__main__":
    main()
