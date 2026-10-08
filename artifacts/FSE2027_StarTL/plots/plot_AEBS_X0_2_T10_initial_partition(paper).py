"""Plot the AEBS X0_2 initial-set satisfaction partition at T=10."""

import csv
import os
import sys
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-starv')

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

from StarV.plot_AEBS_X0_2_star_trajectory import (
    OUTPUT_DIR,
    RESULT_FILE,
    compute_reachable_traces,
)
from StarV.spec.dStarTL import ExpandedFormula, GetSatisfactionFraction
from StarV.util.load import load_AEBS_model_dStarTL, load_AEBS_temporal_specs


HORIZON = 10
INITIAL_SET_ID = 2
INITIAL_SET_NAME = 'X0_{}'.format(INITIAL_SET_ID)
OUTPUT_STEM = 'aebs_X0_2_T10_initial_set_partition'
GRID_COLUMNS = 640
GRID_ROWS = 360
SATISFYING_COLOR = '#2ca25f'
VIOLATING_COLOR = '#d73027'
PLOT_STYLE = {
    'figsize': (10.5, 5.5),
    'font.size': 17,
    'axes.labelsize': 16,
    'axes.titlesize': 19,
    'legend.fontsize': 17,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
}


def apply_plot_style():
    """Apply Matplotlib settings while retaining custom layout entries."""
    plt.rcParams.update({
        key: value for key, value in PLOT_STYLE.items()
        if key in plt.rcParams
    })


def load_reported_result(result_file):
    """Read the exact aggregate result used to annotate the figure."""
    with result_file.open('r', encoding='utf-8') as result_stream:
        for line in result_stream:
            if not line.startswith('AEBS'):
                continue
            fields = line.split()
            if (
                    len(fields) == 16
                    and fields[1] == INITIAL_SET_NAME
                    and int(fields[2]) == HORIZON):
                return {
                    'branches': int(fields[4]),
                    'rho_lb': float(fields[5]),
                    'rho_ub': float(fields[6]),
                    'exact_rho_lb': float(fields[7]),
                    'exact_rho_ub': float(fields[8]),
                    'satisfaction_fraction': float(fields[9]),
                    'satisfying_branches': int(fields[10]),
                    'violating_branches': int(fields[11]),
                    'mixed_branches': int(fields[12]),
                }
    raise RuntimeError('missing AEBS X0_2, T=10 result row')


def build_predicate_grid(initial_star):
    """Return predicate-space cell centers and physical-state cell edges."""
    if initial_star.nVars != 2:
        raise RuntimeError('the AEBS partition plot requires two predicate variables')

    alpha_0_edges = np.linspace(
        initial_star.pred_lb[0], initial_star.pred_ub[0], GRID_COLUMNS + 1
    )
    alpha_1_edges = np.linspace(
        initial_star.pred_lb[1], initial_star.pred_ub[1], GRID_ROWS + 1
    )
    alpha_0 = 0.5 * (alpha_0_edges[:-1] + alpha_0_edges[1:])
    alpha_1 = 0.5 * (alpha_1_edges[:-1] + alpha_1_edges[1:])
    alpha_0_grid, alpha_1_grid = np.meshgrid(alpha_0, alpha_1)
    samples = np.column_stack((alpha_0_grid.ravel(), alpha_1_grid.ravel()))

    distance_edges = initial_star.V[0, 0] + initial_star.V[0, 1:] @ np.vstack((
        alpha_0_edges,
        np.full_like(alpha_0_edges, initial_star.pred_lb[1]),
    ))
    speed_edges = initial_star.V[1, 0] + initial_star.V[1, 1:] @ np.vstack((
        np.full_like(alpha_1_edges, initial_star.pred_lb[0]),
        alpha_1_edges,
    ))
    return samples, distance_edges, speed_edges


def feasible_mask(star, samples, tolerance=1e-9):
    """Test whether predicate samples belong to one coherent final branch."""
    bounds_mask = np.all(
        (samples >= np.asarray(star.pred_lb) - tolerance)
        & (samples <= np.asarray(star.pred_ub) + tolerance),
        axis=1,
    )
    if len(star.C) == 0:
        return bounds_mask
    return bounds_mask & np.all(
        star.C @ samples.T <= star.d[:, np.newaxis] + tolerance,
        axis=0,
    )


def evaluate_partition(traces, specification, initial_star, samples):
    """Evaluate the temporal formula over every feasible ReLU branch."""
    covered = np.zeros(samples.shape[0], dtype=bool)
    satisfying = np.zeros(samples.shape[0], dtype=bool)
    violating = np.zeros(samples.shape[0], dtype=bool)

    for trace in traces:
        expanded_formula = ExpandedFormula(specification, T=len(trace))
        checker = GetSatisfactionFraction(
            trace,
            expanded_formula,
            method='exact',
            lp_solver='linprog',
        )
        branch_mask = feasible_mask(trace[-1], samples)
        if not np.any(branch_mask):
            continue
        branch_satisfaction = checker.evaluateFormulaVolumeSampling(
            initial_star,
            expanded_formula.expr,
            samples,
        )
        covered |= branch_mask
        satisfying |= branch_mask & branch_satisfaction
        violating |= branch_mask & np.logical_not(branch_satisfaction)

    uncovered = np.logical_not(covered)
    conflicts = satisfying & violating
    partition = np.full(samples.shape[0], np.nan)
    partition[covered & np.logical_not(satisfying)] = 0.0
    partition[satisfying] = 1.0
    return partition, uncovered, conflicts


def save_summary(output_file, reported, grid_fraction, uncovered, conflicts):
    """Save reproducibility information for the generated partition."""
    with output_file.open('w', newline='', encoding='utf-8') as csv_stream:
        writer = csv.writer(csv_stream)
        writer.writerow(['quantity', 'value'])
        writer.writerow(['reported_exact_satisfaction_fraction', reported['satisfaction_fraction']])
        writer.writerow(['grid_satisfaction_fraction', grid_fraction])
        writer.writerow(['grid_uncovered_fraction', float(np.mean(uncovered))])
        writer.writerow(['grid_conflict_fraction', float(np.mean(conflicts))])
        writer.writerow(['branches', reported['branches']])
        writer.writerow(['satisfying_branches', reported['satisfying_branches']])
        writer.writerow(['violating_branches', reported['violating_branches']])
        writer.writerow(['mixed_branches', reported['mixed_branches']])


def plot_partition(
        partition, distance_edges, speed_edges, reported, grid_fraction,
        output_dir):
    """Create the initial-state partition and exact fraction summary."""
    apply_plot_style()
    figure, (partition_axis, fraction_axis) = plt.subplots(
        1,
        2,
        figsize=PLOT_STYLE['figsize'],
        gridspec_kw={'width_ratios': [4.5, 1.25]},
    )

    partition_image = partition.reshape(GRID_ROWS, GRID_COLUMNS)
    partition_axis.pcolormesh(
        distance_edges,
        speed_edges,
        partition_image,
        cmap=ListedColormap([VIOLATING_COLOR, SATISFYING_COLOR]),
        vmin=0.0,
        vmax=1.0,
        shading='flat',
        rasterized=True,
    )
    partition_axis.contour(
        0.5 * (distance_edges[:-1] + distance_edges[1:]),
        0.5 * (speed_edges[:-1] + speed_edges[1:]),
        partition_image,
        levels=[0.5],
        colors='black',
        linewidths=1.0,
    )
    partition_axis.add_patch(Rectangle(
        (distance_edges[0], speed_edges[0]),
        distance_edges[-1] - distance_edges[0],
        speed_edges[-1] - speed_edges[0],
        fill=False,
        edgecolor='black',
        linewidth=1.1,
    ))
    partition_axis.set_xlabel(r'Initial distance $d_0$')
    partition_axis.set_ylabel(r'Initial ego speed $v_{\mathrm{ego},0}$')
    partition_axis.set_xlim(distance_edges[0], distance_edges[-1])
    partition_axis.set_ylim(speed_edges[0], speed_edges[-1])
    partition_axis.grid(True, linestyle=':', linewidth=0.5, alpha=0.45)
    partition_axis.legend(
        handles=[
            Patch(facecolor=SATISFYING_COLOR, label=r'Satisfies $\varphi$'),
            Patch(facecolor=VIOLATING_COLOR, label=r'Violates $\varphi$'),
        ],
        loc='lower left',
        framealpha=0.92,
    )
    partition_axis.text(
        0.98,
        0.97,
        (
            r'$X_0^2=[48,48.5]\times[30.2,30.4]$' '\n'
            r'$T=10$, '
            r'$(N_{\mathrm{sat}},N_{\mathrm{viol}},N_{\mathrm{mix}})'
            r'=(%d,%d,%d)$'
        ) % (
            reported['satisfying_branches'],
            reported['violating_branches'],
            reported['mixed_branches'],
        ),
        transform=partition_axis.transAxes,
        ha='right',
        va='top',
        bbox={'facecolor': 'white', 'edgecolor': '0.75', 'alpha': 0.9},
    )

    exact_fraction = reported['satisfaction_fraction']
    fraction_axis.bar(
        0,
        exact_fraction,
        color=SATISFYING_COLOR,
        width=0.62,
        label='Satisfying',
    )
    fraction_axis.bar(
        0,
        1.0 - exact_fraction,
        bottom=exact_fraction,
        color=VIOLATING_COLOR,
        width=0.62,
        label='Violating',
    )
    fraction_axis.text(
        0,
        exact_fraction / 2.0,
        '{:.4f}'.format(exact_fraction),
        color='black',
        weight='bold',
        ha='center',
        va='center',
    )
    fraction_axis.text(
        0,
        exact_fraction + (1.0 - exact_fraction) / 2.0,
        '{:.4f}'.format(1.0 - exact_fraction),
        color='black',
        weight='bold',
        ha='center',
        va='center',
    )
    fraction_axis.set_xlim(-0.65, 0.65)
    fraction_axis.set_ylim(0.0, 1.0)
    fraction_axis.set_xticks([0])
    fraction_axis.set_xticklabels([r'$q_{\varphi}$'])
    fraction_axis.set_ylabel('Predicate-space fraction')
    fraction_axis.grid(True, axis='y', linestyle=':', linewidth=0.5, alpha=0.45)
    fraction_axis.spines['top'].set_visible(False)
    fraction_axis.spines['right'].set_visible(False)
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.22, top=0.97, wspace=0.32)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_file = output_dir / '{}.png'.format(OUTPUT_STEM)
    pdf_file = output_dir / '{}.pdf'.format(OUTPUT_STEM)
    figure.savefig(png_file, dpi=350, bbox_inches='tight')
    figure.savefig(pdf_file, bbox_inches='tight')
    plt.close(figure)
    return png_file, pdf_file


def main():
    reported = load_reported_result(RESULT_FILE)
    _, _, _, _, _, initial_sets = load_AEBS_model_dStarTL()
    initial_star = initial_sets[INITIAL_SET_ID]
    samples, distance_edges, speed_edges = build_predicate_grid(initial_star)
    traces = compute_reachable_traces(HORIZON)
    if len(traces) != reported['branches']:
        raise RuntimeError(
            'reachability returned {} branches, but the result table reports {}'
            .format(len(traces), reported['branches'])
        )

    specification = load_AEBS_temporal_specs()[0]
    partition, uncovered, conflicts = evaluate_partition(
        traces,
        specification,
        initial_star,
        samples,
    )
    covered = np.logical_not(uncovered)
    grid_fraction = float(np.mean(partition[covered] == 1.0))
    if np.mean(uncovered) > 1e-3:
        raise RuntimeError(
            '{:.3%} of the initial-set grid is not covered by any branch'
            .format(np.mean(uncovered))
        )

    png_file, pdf_file = plot_partition(
        partition,
        distance_edges,
        speed_edges,
        reported,
        grid_fraction,
        OUTPUT_DIR,
    )
    summary_file = OUTPUT_DIR / '{}_summary.csv'.format(OUTPUT_STEM)
    save_summary(summary_file, reported, grid_fraction, uncovered, conflicts)
    print('Saved {}'.format(png_file))
    print('Saved {}'.format(pdf_file))
    print('Saved {}'.format(summary_file))
    print('Exact q_phi = {:.6f}; grid check = {:.6f}'.format(
        reported['satisfaction_fraction'], grid_fraction
    ))


if __name__ == '__main__':
    main()
