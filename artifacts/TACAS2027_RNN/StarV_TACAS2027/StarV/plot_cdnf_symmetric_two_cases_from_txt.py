#!/usr/bin/env python3
"""
Symmetric CDNF bar plot for two case studies:
  - LIMO bars on positive y-axis
  - CMAPSS bars on negative y-axis

Default setup:
  - p_f = 0.0
  - T in [10, 15, 20]
  - specs = [phi1, phi3, phi5, phi6] => spec indices [0, 2, 4, 5]

Output:
  - one symmetric figure
  - one parsed CSV containing plotted values
"""

import argparse
import os
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


RUN_RE = re.compile(
    r"^=+\s*(LIMO|CMAPSS|CMPASS) verification:\s*T=(\d+),\s*p_f=([0-9eE+\-.]+)\s*=+"
)
SPEC_RE = re.compile(r"^=+Verify Spec\s+(\d+)\s+\((.+)\)=+")
CDNF_RE = re.compile(r"^Length of Computable DNF\s*=\s*(\d+)\s*for branch signal")

# Requested display renaming:
# phi3 -> phi2, phi5 -> phi3', phi6 -> phi4
DISPLAY_SPEC_LABELS: Dict[int, str] = {
    0: r"$\varphi_1$",
    2: r"$\varphi_2$",
    4: r"$\varphi_3^\prime$",
    5: r"$\varphi_4$",
}


def normalize_case(case_name: str) -> str:
    case_name = case_name.upper()
    if case_name == "CMPASS":
        return "CMAPSS"
    return case_name


def parse_cdnf_rows(txt_path: str) -> pd.DataFrame:
    """Parse [case, T, p_f, spec_idx, spec_label, cdnf_len] rows from one log file."""
    rows: List[dict] = []
    current_case = None
    current_T = None
    current_pf = None
    current_spec_idx = None
    current_spec_label = None

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()

            m_run = RUN_RE.match(line)
            if m_run:
                current_case = normalize_case(m_run.group(1))
                current_T = int(m_run.group(2))
                current_pf = float(m_run.group(3))
                current_spec_idx = None
                current_spec_label = None
                continue

            m_spec = SPEC_RE.match(line)
            if m_spec:
                current_spec_idx = int(m_spec.group(1))
                current_spec_label = m_spec.group(2).strip()
                continue

            m_cdnf = CDNF_RE.match(line)
            if m_cdnf:
                if (
                    current_case is None
                    or current_T is None
                    or current_pf is None
                    or current_spec_idx is None
                    or current_spec_label is None
                ):
                    continue
                rows.append(
                    {
                        "case": current_case,
                        "T": current_T,
                        "p_f": current_pf,
                        "spec_idx": current_spec_idx,
                        "spec_label": current_spec_label,
                        "cdnf_len": int(m_cdnf.group(1)),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No CDNF rows parsed from {txt_path}")
    return df


def aggregate_cdnf(df: pd.DataFrame, agg: str) -> pd.DataFrame:
    """Aggregate branch-level CDNF rows to one value per (case, T, spec)."""
    key_cols = ["case", "T", "spec_idx", "spec_label"]
    if agg == "first":
        return (
            df.groupby(key_cols, as_index=False)
            .agg(cdnf_value=("cdnf_len", "first"), branch_rows=("cdnf_len", "size"))
            .sort_values(["case", "T", "spec_idx"])
        )
    if agg == "mean":
        return (
            df.groupby(key_cols, as_index=False)
            .agg(cdnf_value=("cdnf_len", "mean"), branch_rows=("cdnf_len", "size"))
            .sort_values(["case", "T", "spec_idx"])
        )
    if agg == "sum":
        return (
            df.groupby(key_cols, as_index=False)
            .agg(cdnf_value=("cdnf_len", "sum"), branch_rows=("cdnf_len", "size"))
            .sort_values(["case", "T", "spec_idx"])
        )
    raise ValueError(f"Unsupported agg={agg}")


def _fmt_num(v: float) -> str:
    if np.isclose(v, round(v), atol=1e-10):
        return f"{int(round(v))}"
    return f"{v:.4g}"


def _spec_label(spec_idx: int) -> str:
    return DISPLAY_SPEC_LABELS.get(spec_idx, rf"$\varphi_{{{spec_idx + 1}}}$")


def make_symmetric_plot(
    agg_df: pd.DataFrame,
    t_values: List[int],
    spec_indices: List[int],
    pf: float,
    agg: str,
    out_path: str,
) -> pd.DataFrame:
    """
    Create one symmetric bar plot:
      y>0: LIMO
      y<0: CMAPSS
    Returns plotted table as DataFrame.
    """
    rows = []
    for t in t_values:
        for spec_idx in spec_indices:
            sub_l = agg_df[
                (agg_df["case"] == "LIMO") & (agg_df["T"] == t) & (agg_df["spec_idx"] == spec_idx)
            ]
            sub_c = agg_df[
                (agg_df["case"] == "CMAPSS") & (agg_df["T"] == t) & (agg_df["spec_idx"] == spec_idx)
            ]
            v_l = float(sub_l.iloc[0]["cdnf_value"]) if not sub_l.empty else np.nan
            v_c = float(sub_c.iloc[0]["cdnf_value"]) if not sub_c.empty else np.nan
            rows.append(
                {
                    "T": t,
                    "spec_idx": spec_idx,
                    "spec_label": _spec_label(spec_idx),
                    "LIMO_cdnf": v_l,
                    "CMAPSS_cdnf": v_c,
                }
            )

    plot_df = pd.DataFrame(rows)

    n_t = len(t_values)
    n_spec = len(spec_indices)

    # Evenly spaced T groups with symmetric spec offsets inside each group.
    group_centers = np.arange(n_t, dtype=float)
    intra_spacing = 0.22
    bar_width = 0.16
    t_to_center = {t: group_centers[i] for i, t in enumerate(t_values)}
    spec_offsets = {
        spec_idx: (i - (n_spec - 1) / 2.0) * intra_spacing
        for i, spec_idx in enumerate(spec_indices)
    }
    x_positions = []
    for t in t_values:
        for spec_idx in spec_indices:
            x_positions.append(t_to_center[t] + spec_offsets[spec_idx])

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf"]
    spec_colors = {spec_idx: colors[i % len(colors)] for i, spec_idx in enumerate(spec_indices)}

    # Slightly narrower figure for a more compact layout.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor("#ececec")

    max_abs = 0.0
    for x, (_, row) in zip(x_positions, plot_df.iterrows()):
        t = int(row["T"])
        spec_idx = int(row["spec_idx"])
        color = spec_colors[spec_idx]
        v_l = row["LIMO_cdnf"]
        v_c = row["CMAPSS_cdnf"]

        if not np.isnan(v_l):
            ax.bar(x, v_l, width=bar_width, color=color, edgecolor="black", linewidth=0.6)
            ax.annotate(
                _fmt_num(v_l),
                xy=(x, v_l),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
            max_abs = max(max_abs, abs(float(v_l)))

        if not np.isnan(v_c):
            ax.bar(
                x,
                -v_c,
                width=bar_width,
                color=color,
                edgecolor="black",
                linewidth=0.6,
                hatch="//",
                alpha=0.9,
            )
            ax.annotate(
                _fmt_num(v_c),
                xy=(x, -v_c),
                xytext=(0, -3),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=8.5,
            )
            max_abs = max(max_abs, abs(float(v_c)))

    ax.axhline(0.0, color="black", linewidth=1.0)

    # Group separators between T groups.
    for ti in range(1, n_t):
        sep_x = (group_centers[ti - 1] + group_centers[ti]) / 2.0
        ax.axvline(sep_x, color="gray", linestyle=":", linewidth=0.9, alpha=0.8)

    # X ticks at group centers -> show T only.
    centers = group_centers.tolist()
    ax.set_xticks(centers)
    ax.set_xticklabels([f"T={t}" for t in t_values], fontsize=11)
    if n_t > 0:
        ax.set_xlim(group_centers[0] - 0.6, group_centers[-1] + 0.6)

    ax.set_xlabel(r"$T$ (number of steps)", fontsize=12)
    ax.set_ylabel(r"CDNF Length ", fontsize=12)
    # ax.set_title(
        # rf"Symmetric CDNF Comparison ($p_f={pf:g}$, agg={agg})",
        # fontsize=12,
    # )
    ax.tick_params(axis="y", labelsize=10)

    if max_abs > 0:
        ylim = max_abs * 1.2 + 1
        ax.set_ylim(-ylim, ylim)

    spec_handles = [
        Patch(facecolor=spec_colors[idx], edgecolor="black", label=_spec_label(idx))
        for idx in spec_indices
    ]
    case_handles = [
        Patch(facecolor="white", edgecolor="black", label="LIMO (+)"),
        Patch(facecolor="white", edgecolor="black", hatch="//", label="CMAPSS (-)"),
    ]

    legend1 = ax.legend(
        handles=spec_handles,
        title="Specs",
        fontsize=9,
        title_fontsize=9,
        loc="upper left",
        frameon=True,
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=case_handles,
        title="Cases",
        fontsize=9,
        title_fontsize=9,
        loc="lower left",
        frameon=True,
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return plot_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create symmetric CDNF plot with LIMO on positive and CMAPSS on negative side."
    )
    parser.add_argument(
        "--limo-txt",
        type=str,
        default="/home/qliu1/Desktop/StarV_qing/StarV/LIMO_results_normal_noise.txt",
        help="Path to LIMO verification txt.",
    )
    parser.add_argument(
        "--cmapss-txt",
        type=str,
        default="/home/qliu1/Desktop/StarV_qing/StarV/CMAPSS_results_seed.txt",
        help="Path to CMAPSS verification txt.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/home/qliu1/Desktop/StarV_qing/StarV/results_two_cases/cdnf_vs_T",
        help="Output directory.",
    )
    parser.add_argument(
        "--pf",
        type=float,
        default=0.0,
        help="Filter p_f value (default: 0.0).",
    )
    parser.add_argument(
        "--t-values",
        type=int,
        nargs="+",
        default=[10, 15, 20],
        help="T values to include.",
    )
    parser.add_argument(
        "--spec-idx",
        type=int,
        nargs="+",
        default=[0, 2, 4, 5],
        help="Spec indices to include.",
    )
    parser.add_argument(
        "--agg",
        type=str,
        choices=["first", "mean", "sum"],
        default="first",
        help="Aggregation mode for branch-level CDNF rows.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df_limo = parse_cdnf_rows(args.limo_txt)
    df_cmapss = parse_cdnf_rows(args.cmapss_txt)
    df = pd.concat([df_limo, df_cmapss], ignore_index=True)
    df["case"] = df["case"].apply(normalize_case)
    df = df[df["case"].isin(["LIMO", "CMAPSS"])].copy()

    df = df[np.isclose(df["p_f"].to_numpy(dtype=float), float(args.pf), atol=1e-12)].copy()
    df = df[df["T"].isin(args.t_values)].copy()
    df = df[df["spec_idx"].isin(args.spec_idx)].copy()

    if df.empty:
        raise RuntimeError("No rows after filtering. Check --pf / --t-values / --spec-idx.")

    agg_df = aggregate_cdnf(df, agg=args.agg)

    fig_path = os.path.join(
        args.out_dir,
        f"cdnf_symmetric_two_cases_pf_{args.pf:g}_agg_{args.agg}.png",
    )
    plot_df = make_symmetric_plot(
        agg_df=agg_df,
        t_values=args.t_values,
        spec_indices=args.spec_idx,
        pf=args.pf,
        agg=args.agg,
        out_path=fig_path,
    )

    csv_path = os.path.join(args.out_dir, "parsed_cdnf_symmetric_two_cases.csv")
    plot_df.to_csv(csv_path, index=False)

    print(f"Saved parsed CSV: {csv_path}")
    print(f"Saved figure: {fig_path}")
    print(f"Rows plotted: {len(plot_df)}")


if __name__ == "__main__":
    main()
