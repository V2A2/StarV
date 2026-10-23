#!/usr/bin/env python3
"""
Plot total number of branches vs T at a fixed p_f for two case studies (LIMO and CMAPSS).

Input logs are expected to contain blocks like:
  ======================= <CASE> verification: T=..., p_f=... =======================
  ...
  total branches after reachability:<N>

Output:
  - grouped bar figure with two bars at each T (LIMO, CMAPSS)
  - parsed CSV used for plotting
"""

import argparse
import os
import re
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_RE = re.compile(
    r"^=+\s*(LIMO|CMAPSS|CMPASS) verification:\s*T=(\d+),\s*p_f=([0-9eE+\-.]+)\s*=+"
)
TOTAL_BRANCH_RE = re.compile(r"^total branches after reachability:(\d+)")


def parse_total_branches(txt_path: str, expected_case: str) -> pd.DataFrame:
    """Parse rows [case, T, p_f, total_branches] from one log file."""
    rows = []
    current_case = None
    current_T = None
    current_pf = None

    expected_case = expected_case.upper()

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()

            m_run = RUN_RE.match(line)
            if m_run:
                current_case = m_run.group(1).upper()
                current_T = int(m_run.group(2))
                current_pf = float(m_run.group(3))
                continue

            m_total = TOTAL_BRANCH_RE.match(line)
            if m_total and current_case is not None and current_T is not None and current_pf is not None:
                rows.append(
                    {
                        "case": current_case,
                        "T": current_T,
                        "p_f": current_pf,
                        "total_branches": int(m_total.group(1)),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No total-branch records found in {txt_path}")

    # Normalize typo variant if it appears.
    df["case"] = df["case"].replace({"CMPASS": "CMAPSS"})
    df = df[df["case"] == expected_case].copy()
    if df.empty:
        raise RuntimeError(f"No rows for case {expected_case} found in {txt_path}")
    return df


def filter_pf_and_T(df: pd.DataFrame, pf: float, t_min: int, t_max: int, tol: float = 1e-12) -> pd.DataFrame:
    """Keep one p_f and T range [t_min, t_max]."""
    out = df[(df["p_f"].apply(lambda x: abs(float(x) - pf) <= tol)) & (df["T"] >= t_min) & (df["T"] <= t_max)].copy()
    if out.empty:
        raise RuntimeError(
            f"No rows after filtering with p_f={pf:g}, T in [{t_min}, {t_max}] for case {df['case'].iloc[0]}"
        )
    return out


def plot_two_case_bars(df: pd.DataFrame, out_path: str, pf: float) -> None:
    """Grouped bar chart: at each T there are two bars (LIMO and CMAPSS)."""
    pivot = (
        df.pivot_table(index="T", columns="case", values="total_branches", aggfunc="first")
        .sort_index()
    )

    # Ensure expected case order in plot.
    for c in ["LIMO", "CMAPSS"]:
        if c not in pivot.columns:
            pivot[c] = np.nan
    pivot = pivot[["LIMO", "CMAPSS"]]

    Ts: List[int] = pivot.index.tolist()
    x = np.arange(len(Ts))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor("#ececec")

    bars_limo = ax.bar(
        x - width / 2,
        pivot["LIMO"].values,
        width=width,
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.6,
        label="LIMO",
    )
    bars_cmapss = ax.bar(
        x + width / 2,
        pivot["CMAPSS"].values,
        width=width,
        color="#d62728",
        edgecolor="black",
        linewidth=0.6,
        label="CMAPSS",
    )

    # Annotate exact branch counts on top of each bar.
    def _annotate_bars(container) -> None:
        for rect in container:
            h = rect.get_height()
            if np.isnan(h):
                continue
            ax.annotate(
                f"{int(h)}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    _annotate_bars(bars_limo)
    _annotate_bars(bars_cmapss)

    valid_heights = [h for h in np.r_[pivot["LIMO"].values, pivot["CMAPSS"].values] if not np.isnan(h)]
    if valid_heights:
        ymax = max(valid_heights)
        ax.set_ylim(0, ymax * 1.12 + 1)

    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in Ts], fontsize=11)
    ax.set_xlabel(r"$T$ (number of steps)", fontsize=12)
    ax.set_ylabel("Total Number of Branches", fontsize=12)
    # ax.set_title(rf"Branches vs $T$ for LIMO and CMAPSS ($p_f={pf:g}$)", fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.legend(fontsize=11, loc="best")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot branches vs T at fixed p_f for two case studies (LIMO and CMAPSS)."
    )
    parser.add_argument(
        "--limo-txt",
        type=str,
        default="/home/qliu1/Desktop/StarV_qing/StarV/LIMO_results_normal_noise.txt",
        help="Path to LIMO txt log.",
    )
    parser.add_argument(
        "--cmapss-txt",
        type=str,
        default="/home/qliu1/Desktop/StarV_qing/StarV/CMAPSS_results_seed.txt",
        help="Path to CMAPSS txt log.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/home/qliu1/Desktop/StarV_qing/StarV/results_two_cases/branches_vs_T",
        help="Output directory for figure and parsed CSV.",
    )
    parser.add_argument(
        "--pf",
        type=float,
        default=0.0,
        help="Filter value p_f to plot (default: 0.0).",
    )
    parser.add_argument(
        "--t-min",
        type=int,
        default=10,
        help="Minimum T (inclusive).",
    )
    parser.add_argument(
        "--t-max",
        type=int,
        default=25,
        help="Maximum T (inclusive).",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df_limo = parse_total_branches(args.limo_txt, expected_case="LIMO")
    df_cmapss = parse_total_branches(args.cmapss_txt, expected_case="CMAPSS")

    df_limo = filter_pf_and_T(df_limo, pf=args.pf, t_min=args.t_min, t_max=args.t_max)
    df_cmapss = filter_pf_and_T(df_cmapss, pf=args.pf, t_min=args.t_min, t_max=args.t_max)

    df_plot = pd.concat([df_limo, df_cmapss], ignore_index=True)
    df_plot = df_plot.sort_values(["T", "case"]).reset_index(drop=True)

    csv_path = os.path.join(args.out_dir, "parsed_branches_two_cases_pf.csv")
    df_plot.to_csv(csv_path, index=False)

    fig_path = os.path.join(args.out_dir, f"branches_two_cases_pf_{args.pf:g}_T_{args.t_min}_{args.t_max}.png")
    plot_two_case_bars(df=df_plot, out_path=fig_path, pf=args.pf)

    print(f"Saved parsed CSV: {csv_path}")
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
