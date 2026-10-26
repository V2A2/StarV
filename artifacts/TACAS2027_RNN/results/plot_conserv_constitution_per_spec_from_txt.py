#!/usr/bin/env python3
"""
Generate per-spec conservativeness/constitution figures from a LIMO verification log.


"""

import argparse
import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


RUN_RE = re.compile(
    r"^=+\s*(?:CMAPSS|LIMO) verification:\s*T=(\d+),\s*p_f=([0-9eE+\-.]+)\s*=+"
)
SPEC_RE = re.compile(r"^=+Verify Spec\s+(\d+)\s+\((.+)\)=+")
CONSERV_RE = re.compile(r"^conservativeness for Spec\s+(\d+):\s*([0-9eE+\-.]+)%")
CONSTIT_RE = re.compile(r"^constitution for Spec\s+(\d+):\s*([0-9eE+\-.]+)%")


def parse_limo_conserv_rows(txt_path: str) -> pd.DataFrame:
    """Parse rows: [T, p_f, spec_idx, spec_label, conserv_list, consist_list]."""
    rows: List[dict] = []

    current_T = None
    current_pf = None
    current_spec_idx = None
    current_spec_label = None
    pending_conserv: Dict[Tuple[int, float, int], float] = {}

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()

            m_run = RUN_RE.match(line)
            if m_run:
                current_T = int(m_run.group(1))
                current_pf = float(m_run.group(2))
                current_spec_idx = None
                current_spec_label = None
                continue

            m_spec = SPEC_RE.match(line)
            if m_spec:
                current_spec_idx = int(m_spec.group(1))
                current_spec_label = m_spec.group(2).strip()
                continue

            m_conserv = CONSERV_RE.match(line)
            if m_conserv:
                idx = int(m_conserv.group(1))
                val = float(m_conserv.group(2))
                if (
                    current_T is None
                    or current_pf is None
                    or current_spec_idx is None
                    or current_spec_label is None
                    or idx != current_spec_idx
                ):
                    continue
                key = (current_T, current_pf, idx)
                pending_conserv[key] = val
                continue

            m_constit = CONSTIT_RE.match(line)
            if m_constit:
                idx = int(m_constit.group(1))
                val = float(m_constit.group(2))
                if (
                    current_T is None
                    or current_pf is None
                    or current_spec_idx is None
                    or current_spec_label is None
                    or idx != current_spec_idx
                ):
                    continue
                key = (current_T, current_pf, idx)
                if key not in pending_conserv:
                    continue

                rows.append(
                    {
                        "T": current_T,
                        "p_f": current_pf,
                        "spec_idx": idx,
                        "spec_label": current_spec_label,
                        "conserv_list": pending_conserv[key],
                        "consist_list": val,
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No conserv/constitution rows parsed from {txt_path}")
    return df


def sanitize_spec_tag(spec_label: str, fallback_idx: int) -> str:
    match = re.search(r"\d+", str(spec_label))
    if match:
        return f"spec_{match.group(0)}"
    return f"spec_{fallback_idx + 1}"


def format_pf_label(pf: float) -> str:
    s = f"{pf:.12g}"
    if "." not in s and "e" not in s.lower():
        s += ".0"
    return s


def plot_conserv_constitution_vs_T_per_spec_from_txt(
    txt_path: str,
    out_dir: str,
    exclude_T: Tuple[int, ...] = (30,),
    include_p_filters: Tuple[float, ...] = (0.0, 0.01, 0.1),
) -> Tuple[List[str], str]:
    """
    Create one 1x2 figure per specification:
      Left: conservativeness vs T (curves for p_f)
      Right: constitution vs T (curves for p_f)
    """
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures_conserv_constitution_per_spec")
    os.makedirs(fig_dir, exist_ok=True)

    df = parse_limo_conserv_rows(txt_path)
    if exclude_T:
        df = df[~df["T"].isin(list(exclude_T))].copy()
    if df.empty:
        raise RuntimeError("No rows remain after T filtering.")
    if include_p_filters:
        mask = df["p_f"].apply(lambda x: any(abs(float(x) - pf) < 1e-12 for pf in include_p_filters))
        df = df[mask].copy()
    if df.empty:
        raise RuntimeError(
            f"No rows remain after p_f filtering. Requested p_f values: {include_p_filters}"
        )

    parsed_csv = os.path.join(out_dir, "parsed_limo_conserv_constit_rows.csv")
    df.sort_values(["spec_idx", "T", "p_f"]).to_csv(parsed_csv, index=False)

    markers = ["o", "x", ">", "s", "d", "^", "v", "P", "*"]
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["blue", "red", "#1f77b4"])

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

        p_filters_sorted = sorted(df_spec["p_f"].dropna().unique().tolist())

        fig, (ax_cons, ax_constit) = plt.subplots(1, 2, figsize=(14, 5))
        ax_cons.set_facecolor("#ececec")
        ax_constit.set_facecolor("#ececec")

        for j, pf in enumerate(p_filters_sorted):
            df_curve = (
                df_spec[df_spec["p_f"] == pf]
                .groupby("T", as_index=False)[["conserv_list", "consist_list"]]
                .mean()
                .sort_values("T")
            )
            if df_curve.empty:
                continue

            color = colors[j % len(colors)]
            marker = markers[j % len(markers)]
            label = rf"$p_f = {format_pf_label(pf)}$"

            ax_cons.plot(
                df_curve["T"],
                df_curve["conserv_list"],
                color=color,
                marker=marker,
                linewidth=1.2,
                markersize=6,
                label=label,
            )
            ax_constit.plot(
                df_curve["T"],
                df_curve["consist_list"],
                color=color,
                marker=marker,
                linewidth=1.2,
                markersize=6,
                label=label,
            )

        ax_cons.set_xlabel(r"$T$ (number of steps)", fontsize=13)
        ax_cons.set_ylabel("Conservativeness (%)", fontsize=13)
        ax_cons.tick_params(axis="both", labelsize=11)
        ax_cons.legend(fontsize=12, loc="best")

        ax_constit.set_xlabel(r"$T$ (number of steps)", fontsize=13)
        ax_constit.set_ylabel("Constitution (%)", fontsize=13)
        ax_constit.tick_params(axis="both", labelsize=11)
        ax_constit.legend(fontsize=12, loc="best")

        plt.tight_layout()
        spec_tag = sanitize_spec_tag(spec_label, fallback_idx=i)
        fig_path = os.path.join(fig_dir, f"limo_{spec_tag}_conserv_constitution_vs_T.png")
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved conserv/constitution figure: {fig_path}")
        figure_paths.append(fig_path)

    return figure_paths, parsed_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-spec conservativeness/constitution-vs-T figures from LIMO_results_normal_noise.txt"

        # description="Generate per-spec conservativeness/constitution-vs-T figures from CMAPSS_results_seed.txt"
    )
    parser.add_argument(
        "--txt",
        type=str,
        # default="/home/qliu1/Desktop/StarV_qing/StarV/CMAPSS_results_seed.txt",
        default="/home/qliu1/Desktop/StarV_qing/StarV/LIMO_results_normal_noise.txt",
        help="Path to LIMO results txt file.",
    )
    parser.add_argument(
        "--out-dir",
        # type=str,
        # default="/home/qliu1/Desktop/StarV_qing/StarV/results_CMAPSS/figures_conserv_constitution_per_spec_all_pf",

        default="/home/qliu1/Desktop/StarV_qing/StarV/results_LIMO/figures_conserv_constitution_per_spec_all_pf",
        # help="Output directory (figure folder will be created inside).",
    )
    parser.add_argument(
        "--exclude-T",
        type=int,
        nargs="*",
        default=[25,30],
        help="T values to exclude (default: 25).",
    )
    parser.add_argument(
        "--include-pf",
        type=float,
        nargs="*",
        default=[0.0,0.005,0.01,0.05,0.1],
        help="Only plot these p_f values (default: 0.0 0.01 0.1).",
    )
    args = parser.parse_args()

    figs, parsed_csv = plot_conserv_constitution_vs_T_per_spec_from_txt(
        txt_path=args.txt,
        out_dir=args.out_dir,
        exclude_T=tuple(args.exclude_T),
        include_p_filters=tuple(args.include_pf),
    )
    print(f"Saved parsed CSV: {parsed_csv}")
    print(f"Saved figures: {len(figs)}")


if __name__ == "__main__":
    main()
