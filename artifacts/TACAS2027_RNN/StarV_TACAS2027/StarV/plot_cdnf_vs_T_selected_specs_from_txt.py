#!/usr/bin/env python3
"""
Plot CDNF length vs T for selected temporal specs from a verification txt log.

Default plot setup:
  - T values: 10, 15, 20
  - specs: phi1, phi3, phi5, phi6  (spec indices 0, 2, 4, 5)
  - p_f filter: 0.0
  - bar labels: exact numeric value on top of each bar

Input log lines used:
  ======================= <CASE> verification: T=..., p_f=... =======================
  ==================Verify Spec <idx> (<label>)====================
  Length of Computable DNF = <N> for branch signal
"""

import argparse
import os
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_RE = re.compile(
    r"^=+\s*(LIMO|CMAPSS|CMPASS) verification:\s*T=(\d+),\s*p_f=([0-9eE+\-.]+)\s*=+"
)
SPEC_RE = re.compile(r"^=+Verify Spec\s+(\d+)\s+\((.+)\)=+")
CDNF_RE = re.compile(r"^Length of Computable DNF\s*=\s*(\d+)\s*for branch signal")

# Custom display names for selected specs in the figure legend.
# Requested mapping:
#   phi3 -> phi2, phi5 -> phi3', phi6 -> phi4
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
    """
    Parse rows:
      [case, T, p_f, spec_idx, spec_label, cdnf_len]

    One row is emitted per "Length of Computable DNF = ..." line.
    """
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
    """
    Aggregate multiple branch-level CDNF rows into one value per (T, spec).
    """
    if agg == "first":
        out = (
            df.groupby(["T", "spec_idx", "spec_label"], as_index=False)
            .agg(cdnf_value=("cdnf_len", "first"), branch_rows=("cdnf_len", "size"))
            .sort_values(["T", "spec_idx"])
        )
        return out
    if agg == "mean":
        out = (
            df.groupby(["T", "spec_idx", "spec_label"], as_index=False)
            .agg(cdnf_value=("cdnf_len", "mean"), branch_rows=("cdnf_len", "size"))
            .sort_values(["T", "spec_idx"])
        )
        return out
    if agg == "sum":
        out = (
            df.groupby(["T", "spec_idx", "spec_label"], as_index=False)
            .agg(cdnf_value=("cdnf_len", "sum"), branch_rows=("cdnf_len", "size"))
            .sort_values(["T", "spec_idx"])
        )
        return out
    raise ValueError(f"Unsupported agg={agg}")


def _label_for_value(v: float) -> str:
    if np.isclose(v, round(v), atol=1e-10):
        return f"{int(round(v))}"
    return f"{v:.4g}"


def plot_grouped_cdnf(
    agg_df: pd.DataFrame,
    spec_indices: List[int],
    t_values: List[int],
    out_path: str,
    pf: float,
    agg: str,
) -> None:
    """
    Grouped bars:
      x-axis = T
      bars at each T = selected specs
    """
    spec_map: Dict[int, str] = {}
    for spec_idx in spec_indices:
        if spec_idx in DISPLAY_SPEC_LABELS:
            spec_map[spec_idx] = DISPLAY_SPEC_LABELS[spec_idx]
            continue
        sub = agg_df[agg_df["spec_idx"] == spec_idx]
        if sub.empty:
            spec_map[spec_idx] = rf"$\varphi_{{{spec_idx + 1}}}$"
        else:
            spec_map[spec_idx] = str(sub.iloc[0]["spec_label"])

    x = np.arange(len(t_values))
    n_spec = len(spec_indices)
    width = min(0.8 / max(n_spec, 1), 0.22)

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.set_facecolor("#ececec")
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf"]

    max_y = 0.0
    for i, spec_idx in enumerate(spec_indices):
        vals = []
        for t in t_values:
            sub = agg_df[(agg_df["T"] == t) & (agg_df["spec_idx"] == spec_idx)]
            if sub.empty:
                vals.append(np.nan)
            else:
                vals.append(float(sub.iloc[0]["cdnf_value"]))

        offset = (i - (n_spec - 1) / 2.0) * width
        bars = ax.bar(
            x + offset,
            vals,
            width=width * 0.95,
            color=colors[i % len(colors)],
            edgecolor="black",
            linewidth=0.6,
            label=spec_map[spec_idx],
        )

        for bar, val in zip(bars, vals):
            if np.isnan(val):
                continue
            max_y = max(max_y, float(val))
            ax.annotate(
                _label_for_value(float(val)),
                xy=(bar.get_x() + bar.get_width() / 2, float(val)),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in t_values], fontsize=11)
    ax.set_xlabel(r"$T$ (number of steps)", fontsize=12)
    ax.set_ylabel("Number of CDNF", fontsize=12)
    # ax.set_title(rf"CDNF vs $T$ for Selected Specs ($p_f={pf:g}$, agg={agg})", fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.legend(fontsize=11, loc="best")
    if max_y > 0:
        ax.set_ylim(0, max_y * 1.15 + 1)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot CDNF vs T for selected specs from verification txt log."
    )
    parser.add_argument(
        "--txt",
        type=str,
        default="/home/qliu1/Desktop/StarV_qing/StarV/LIMO_results_normal_noise.txt",
        help="Path to verification txt file.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/home/qliu1/Desktop/StarV_qing/StarV/results_two_cases/cdnf_vs_T_LIMO",
        help="Output directory for figure and parsed csv.",
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        choices=[None, "LIMO", "CMAPSS"],
        help="Optional case filter. If omitted, keep all cases in file.",
    )
    parser.add_argument(
        "--pf",
        type=float,
        default=0.0,
        help="p_f value to filter (default: 0.0).",
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
        help="Spec indices to include (default maps to phi1, phi3, phi5, phi6).",
    )
    parser.add_argument(
        "--agg",
        type=str,
        choices=["first", "mean", "sum"],
        default="first",
        help="How to aggregate branch-level CDNF rows into one value per (T, spec).",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = parse_cdnf_rows(args.txt)

    if args.case is not None:
        want_case = normalize_case(str(args.case))
        df = df[df["case"] == want_case].copy()

    # Filter p_f, T, spec set.
    df = df[np.isclose(df["p_f"].to_numpy(dtype=float), float(args.pf), atol=1e-12)].copy()
    df = df[df["T"].isin(args.t_values)].copy()
    df = df[df["spec_idx"].isin(args.spec_idx)].copy()

    if df.empty:
        raise RuntimeError("No rows after filtering. Check --txt / --case / --pf / --t-values / --spec-idx.")

    agg_df = aggregate_cdnf(df, agg=args.agg)
    agg_df = agg_df.sort_values(["T", "spec_idx"]).reset_index(drop=True)

    csv_path = os.path.join(args.out_dir, "parsed_cdnf_selected_specs.csv")
    agg_df.to_csv(csv_path, index=False)

    fig_path = os.path.join(
        args.out_dir,
        f"cdnf_vs_T_pf_{args.pf:g}_agg_{args.agg}.png",
    )
    plot_grouped_cdnf(
        agg_df=agg_df,
        spec_indices=args.spec_idx,
        t_values=args.t_values,
        out_path=fig_path,
        pf=args.pf,
        agg=args.agg,
    )

    print(f"Saved parsed CSV: {csv_path}")
    print(f"Saved figure: {fig_path}")
    print(f"Rows plotted: {len(agg_df)}")


if __name__ == "__main__":
    main()
