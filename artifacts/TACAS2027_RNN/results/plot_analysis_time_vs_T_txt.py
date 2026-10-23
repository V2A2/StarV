#!/usr/bin/env python3
"""
Generate per-spec analysis-time figures from a LIMO verification log file.

Output figures:
  results/TL_verification_benchmark/figures_analysis_time_per_spec/
    limo_spec_<k>_analysis_time_vs_T.png

By default this script plots T values up to 25 and excludes T=30.
Select one pruning threshold in main() or with --pf; different p_f values are never averaged.
Example (from the project root):
  python StarV/plot_analysis_time_vs_T_txt.py --pf 0.005 --max-T 20 \
    --txt StarV/CMAPSS_results_seed.txt --out-dir StarV/results_CMAPSS/with_T20_pf0.005
"""

import argparse
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd



RUN_RE = re.compile(
    r"^=+\s*(?:CMAPSS|LIMO) verification:\s*T=(\d+),\s*p_f=([0-9eE+\-.]+)\s*=+"
)
REACH_RE = re.compile(r"^Reachability analysis time:\s*([0-9eE+\-.]+)\s*seconds")
SPEC_RE = re.compile(r"^=+Verify Spec\s+(\d+)\s+\((.+)\)=+")
CHECK_RE = re.compile(r"^Checking time for Spec\s+(\d+):\s*([0-9eE+\-.]+)\s*seconds")


def parse_limo_analysis_rows(txt_path: str) -> pd.DataFrame:
    """Parse rows: [T, p_f, spec_idx, spec_label, t_r, t_c, t_v]."""
    records: List[dict] = []

    current_T = None
    current_pf = None
    current_t_r = None
    current_spec_idx = None
    current_spec_label = None

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()

            m_run = RUN_RE.match(line)
            if m_run:
                current_T = int(m_run.group(1))
                current_pf = float(m_run.group(2))
                current_t_r = None
                current_spec_idx = None
                current_spec_label = None
                continue

            m_reach = REACH_RE.match(line)
            if m_reach:
                current_t_r = float(m_reach.group(1))
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
                    current_T is None
                    or current_pf is None
                    or current_t_r is None
                    or current_spec_idx is None
                    or current_spec_label is None
                ):
                    continue
                if check_idx != current_spec_idx:
                    continue

                records.append(
                    {
                        "T": current_T,
                        "p_f": current_pf,
                        "spec_idx": current_spec_idx,
                        "spec_label": current_spec_label,
                        "t_r": current_t_r,
                        "t_c": t_c,
                        "t_v": current_t_r + t_c,
                    }
                )

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(f"No analysis-time rows parsed from {txt_path}")
    return df


def sanitize_spec_tag(spec_label: str, fallback_idx: int) -> str:
    match = re.search(r"\d+", str(spec_label))
    if match:
        return f"spec_{match.group(0)}"
    return f"spec_{fallback_idx + 1}"


def plot_analysis_time_vs_T_per_spec_from_txt(
    txt_path: str,
    out_dir: str,
    exclude_T: Tuple[int, ...] = (30,),
    max_T: int = 25,
    pf: Optional[float] = None,
) -> Tuple[List[str], str]:
    """
    Create one 1x1 figure per spec:
      Analysis time vs T at one fixed p_f, with t_r, t_c, t_v.
      Only repeated runs at the same T and p_f are averaged.
    """
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures_analysis_time_per_spec")
    os.makedirs(fig_dir, exist_ok=True)

    df = parse_limo_analysis_rows(txt_path)
    df = df[df["T"] <= max_T].copy()
    if exclude_T:
        df = df[~df["T"].isin(list(exclude_T))].copy()
    if df.empty:
        raise RuntimeError("No rows remain after T filtering.")

    available_pf = sorted(df["p_f"].unique())
    if pf is None or not math.isfinite(pf) or pf < 0:
        raise ValueError(f"Select a finite, nonnegative p_f with --pf. Available values: {available_pf}")
    matches = [value for value in available_pf if math.isclose(value, pf, rel_tol=1e-9, abs_tol=0.0)]
    if len(matches) != 1:
        raise ValueError(f"p_f={pf:g} must match one available value. Available values: {available_pf}")
    expected_pairs = set(zip(df["spec_idx"], df["T"]))
    df = df[df["p_f"] == matches[0]].copy()
    missing_pairs = expected_pairs - set(zip(df["spec_idx"], df["T"]))
    if missing_pairs:
        raise ValueError(f"p_f={pf:g} is missing data for (spec_idx, T): {sorted(missing_pairs)}")
    print(f"Plotting fixed p_f={matches[0]:g}")

    parsed_csv = os.path.join(out_dir, "parsed_limo_analysis_time_rows.csv")
    df.sort_values(["spec_idx", "T", "p_f"]).to_csv(parsed_csv, index=False)

    metric_styles = {
        "t_r": {"label": r"$t_r$", "color": "blue", "marker": "o"},
        "t_c": {"label": r"$t_c$", "color": "red", "marker": "x"},
        "t_v": {"label": r"$t_v$", "color": "#1f77b4", "marker": ">"},
    }

    spec_order = (
        df[["spec_idx", "spec_label"]]
        .drop_duplicates()
        .sort_values("spec_idx")
        .values.tolist()
    )
    figure_paths: List[str] = []

    for i, (spec_idx, spec_label) in enumerate(spec_order):
        df_spec = df[df["spec_idx"] == spec_idx].copy()
        if df_spec.empty:
            continue

        by_T = (
            df_spec.groupby("T", as_index=False)[["t_r", "t_c", "t_v"]]
            .mean()
            .sort_values("T")
        )
        if by_T.empty:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(6.6, 4.4))
        ax.set_facecolor("#ececec")

        for metric, style in metric_styles.items():
            ax.plot(
                by_T["T"],
                by_T[metric],
                color=style["color"],
                marker=style["marker"],
                linewidth=1.2,
                markersize=4,
                label=style["label"],
            )

        ax.set_xlabel(r"$T$ (number of steps)", fontsize=12)
        ax.set_ylabel("Analysis Time (s)", fontsize=12)
        # No title, matching plot_pf_vs_analysis_from_txt.py.
        ax.tick_params(axis="both", labelsize=10)
        ax.legend(fontsize=11, loc="best")

        plt.tight_layout()
        spec_tag = sanitize_spec_tag(spec_label, fallback_idx=i)
        fig_path = os.path.join(fig_dir, f"cmapss_{spec_tag}_analysis_time_vs_T.png")
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved analysis-time figure: {fig_path}")
        figure_paths.append(fig_path)

    return figure_paths, parsed_csv


def main() -> None:
    # Edit these settings, then run this script without command-line arguments.
    selected_pf = 0.0
    selected_max_T = 20  # Choose 20 or 25.
    excluded_T = [30]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(script_dir, "CMAPSS_results_seed.txt")
    output_root = os.path.join(script_dir, "results_CMAPSS")

    # Command-line arguments can optionally override the settings above.
    parser = argparse.ArgumentParser(
        description="Generate per-spec analysis-time-vs-T figures at a fixed p_f"
    )
    parser.add_argument("--txt", type=str, default=txt_path, help="Results log file.")
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory; by default determined by the selected T and p_f.",
    )
    parser.add_argument("--exclude-T", type=int, nargs="*", default=excluded_T)
    parser.add_argument("--max-T", type=int, choices=(20, 25), default=selected_max_T)
    parser.add_argument("--pf", type=float, default=selected_pf, help="Fixed p_f value.")
    args = parser.parse_args()
    if args.max_T not in (20, 25):
        parser.error("selected_max_T must be 20 or 25")
    if args.out_dir is None:
        args.out_dir = os.path.join(output_root, f"with_T{args.max_T}_pf{args.pf:g}")

    figs, parsed_csv = plot_analysis_time_vs_T_per_spec_from_txt(
        txt_path=args.txt,
        out_dir=args.out_dir,
        exclude_T=tuple(args.exclude_T),
        max_T=args.max_T,
        pf=args.pf,
    )
    print(f"Saved parsed CSV: {parsed_csv}")
    print(f"Saved figures: {len(figs)}")


if __name__ == "__main__":
    main()
