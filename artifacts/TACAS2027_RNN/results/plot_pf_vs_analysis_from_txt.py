#!/usr/bin/env python3
"""
Create p_f vs analysis-time figures from a CMPASS verification text log.

Default target specs (conceptual name -> printed spec in the log):
    phi1  -> $\\varphi_1$
    phi2  -> $\\varphi_3$
    phi31 -> $\\varphi_5$   (phi31 is replaced by equivalent phi3 in current runs)
    phi4  -> $\\varphi_6$

For each target spec and each T, this script saves one figure with:
    x-axis: p_f
    curves: t_r, t_c, t_v

It also supports generating per-spec figures directly from summary tables in
the same txt log:
    - figures_analysis_time_per_spec
    - figures_conserv_constitution_per_spec
with optional exclusion of specific T values (e.g., T = 30).
"""

import argparse
import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


RUN_RE = re.compile(
    r"^=+\s*(?:CMAPSS|CMPASS|LIMO) verification:\s*T=(\d+),\s*p_f=([0-9eE+\-.]+)\s*=+"
)
REACH_RE = re.compile(r"^Reachability analysis time:\s*([0-9eE+\-.]+)\s*seconds")
SPEC_RE = re.compile(r"^=+Verify Spec\s+(\d+)\s+\((.+)\)=+")
CHECK_RE = re.compile(r"^Checking time for Spec\s+(\d+):\s*([0-9eE+\-.]+)\s*seconds")
PF_AXIS_ORDER = [0.0, 0.005, 0.01, 0.05, 0.1]
TABLE_BLOCK_RE = re.compile(
    r"^=+\s*VERIFICATION RESULTS (?:CMAPSS|CMPASS|LIMO) WITH FILTERING pf\s*=\s*([0-9eE+\-.]+)\s*=+"
)


def _format_pf_label(pf: float) -> str:
    if abs(pf - round(pf)) < 1e-12:
        return str(int(round(pf)))
    return f"{pf:g}"


def _safe_float(tok: str) -> float:
    t = tok.strip().lower()
    if t == "nan":
        return float("nan")
    return float(tok)


def parse_limo_log(txt_path: str) -> pd.DataFrame:
    """Parse log file into rows: [T, p_f, spec_idx, spec_label, t_r, t_c, t_v]."""
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

                # If file is well-formed these should match.
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
        raise RuntimeError(f"No parseable records found in {txt_path}")
    return df


def parse_limo_summary_tables(txt_path: str) -> pd.DataFrame:
    """
    Parse final summary-table blocks:
      'VERIFICATION RESULTS CMPASS WITH FILTERING pf = ...'
    into a detail-style DataFrame.
    """
    records: List[dict] = []
    current_pf = None

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()

            m_block = TABLE_BLOCK_RE.match(line)
            if m_block:
                current_pf = float(m_block.group(1))
                continue

            if current_pf is None:
                continue

            # Summary rows start with '$\\varphi_i$'
            if not line.startswith("$\\varphi_"):
                continue

            toks = line.split()
            # Expected columns:
            # spec, T, p_max, p_min, reachTime, checkTime, verifyTime,
            # p_ig_check, p_ig_eval, conserv(%), constit(%)
            if len(toks) < 11:
                continue

            spec_name = toks[0]
            T = int(float(toks[1]))
            rho_max = _safe_float(toks[2])
            rho_min = _safe_float(toks[3])
            t_r = _safe_float(toks[4])
            t_c = _safe_float(toks[5])
            t_v = _safe_float(toks[6])
            p_ig_check = _safe_float(toks[7])
            p_ig_eval = _safe_float(toks[8])
            conserv = _safe_float(toks[9])
            consist = _safe_float(toks[10])

            records.append(
                {
                    "case": "CMAPSS",  # or "CMPASS" if parsing from CMPASS logs
                    "T": T,
                    "p_filter": current_pf,
                    "spec_name": spec_name,
                    "rho_max": rho_max,
                    "rho_min": rho_min,
                    "t_r": t_r,
                    "t_c": t_c,
                    "t_v": t_v,
                    "p_ig_check": p_ig_check,
                    "p_ig_eval": p_ig_eval,
                    "conserv_list": conserv,
                    "consist_list": consist,
                }
            )

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(f"No summary-table rows found in {txt_path}")
    return df


def make_pf_plot(
    df_sub: pd.DataFrame,
    title: str,
    out_path: str,
    face_color: str = "#ececec",
) -> None:
    """Create one p_f vs analysis-time panel, matching the requested style."""
    df_sub = df_sub.sort_values("p_f")

    # Use evenly spaced categorical x positions for p_f values.
    pf_values = sorted(df_sub["p_f"].unique().tolist())
    pf_order = [pf for pf in PF_AXIS_ORDER if pf in pf_values]
    for pf in pf_values:
        if pf not in pf_order:
            pf_order.append(pf)
    pf_to_x = {pf: i for i, pf in enumerate(pf_order)}
    df_sub = df_sub.assign(x=df_sub["p_f"].map(pf_to_x)).sort_values("x")

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.set_facecolor(face_color)

    ax.plot(
        df_sub["x"],
        df_sub["t_r"],
        color="blue",
        marker="o",
        markersize=4,
        linewidth=1.2,
        label=r"$t_r$",
    )
    ax.plot(
        df_sub["x"],
        df_sub["t_c"],
        color="red",
        marker="x",
        markersize=4,
        linewidth=1.2,
        label=r"$t_c$",
    )
    ax.plot(
        df_sub["x"],
        df_sub["t_v"],
        color="#1f77b4",
        marker=">",
        markersize=4,
        linewidth=1.2,
        label=r"$t_v$",
    )

    # ax.set_title(title, fontsize=11)
    ax.set_xlabel(r"$p_f$", fontsize=12)
    ax.set_ylabel("Analysis Time (s)", fontsize=12)
    ax.set_xticks(list(range(len(pf_order))))
    ax.set_xticklabels([f"{pf:g}" for pf in pf_order])
    if pf_order:
        ax.set_xlim(-0.2, len(pf_order) - 0.8)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=11, loc="best")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_default_spec_map() -> Dict[str, str]:
    """
    Mapping from requested conceptual spec names to labels in this CMPASS log.

    Current create_specs_CMPASS order in your run:
      [phi1, phi11, phi2, phi21, phi3, phi4] -> [varphi_1 ... varphi_6]
    So:
      phi1  -> varphi_1
      phi2  -> varphi_3
      phi31 -> varphi_5   (using phi3 as equivalent replacement for phi31)
      phi4  -> varphi_6
    """
    return {
        "phi1": r"$\varphi_1$",
        "phi2": r"$\varphi_3$",
        "phi31": r"$\varphi_5$",
        "phi4": r"$\varphi_6$",
    }


def generate_figures(df: pd.DataFrame, out_dir: str) -> Tuple[List[str], str]:
    """
    Save one figure per (target_spec, T), and one CSV with parsed rows.
    Returns: (list_of_figure_paths, csv_path)
    """
    os.makedirs(out_dir, exist_ok=True)
    spec_map = build_default_spec_map()
    Ts = sorted(df["T"].unique().tolist())
    saved_figs: List[str] = []

    # Save parsed data for easy reuse/debug.
    csv_path = os.path.join(out_dir, "parsed_limo_pf_vs_analysis.csv")
    df.sort_values(["spec_label", "T", "p_f"]).to_csv(csv_path, index=False)

    for spec_name, spec_label in spec_map.items():
        df_spec = df[df["spec_label"] == spec_label].copy()
        if df_spec.empty:
            print(f"[WARN] No rows found for {spec_name} ({spec_label}); skipping.")
            continue

        for T in Ts:
            df_sub = df_spec[df_spec["T"] == T].copy()
            if df_sub.empty:
                continue

            fig_name = f"limo_{spec_name}_T{T}_pf_vs_analysis.png"
            fig_path = os.path.join(out_dir, fig_name)
            title = f"CMAPSS {spec_name} ({spec_label}), T={T}"
            make_pf_plot(df_sub=df_sub, title=title, out_path=fig_path)
            saved_figs.append(fig_path)

    return saved_figs, csv_path


def _sanitize_spec_tag(spec_name: str, fallback_idx: int = None) -> str:
    m = re.search(r"\d+", str(spec_name))
    if m:
        return f"spec_{m.group(0)}"
    if fallback_idx is not None:
        return f"spec_{fallback_idx + 1}"
    return "spec"




def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate p_f vs analysis-time figures from CMAPSS verification txt log."
    )
    parser.add_argument(
        "--txt",
        type=str,
        default="/home/qliu1/Desktop/StarV_qing/StarV/CMAPSS_results_seed.txt",
        # default="/home/qliu1/Desktop/StarV_qing/StarV/LIMO_results_normal_noise.txt",
        help="Path to the CMAPSS verification txt file.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/home/qliu1/Desktop/StarV_qing/StarV/results_CMAPSS/pf_vs_analysis_phi_selected",
        # default="/home/qliu1/Desktop/StarV_qing/StarV/results_LIMO/pf_vs_analysis_phi_selected",
        help="Output directory for figures and parsed CSV.",
    )
    args = parser.parse_args()

    df = parse_limo_log(args.txt)
    figs, csv_path = generate_figures(df=df, out_dir=args.out_dir)

    print(f"Parsed rows: {len(df)}")
    print(f"Saved parsed CSV: {csv_path}")
    print(f"Saved figures: {len(figs)}")
    for p in figs:
        print(p)


if __name__ == "__main__":
    main()
