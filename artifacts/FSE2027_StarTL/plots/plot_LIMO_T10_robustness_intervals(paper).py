import csv
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-starv')

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULT_FILE = (
    ROOT
    / 'dstarTL_results'
    / 'test_LIMO_sampling_noise0.0018_T1015202530_new_spec_core1.txt'
)
BRANCH_INTERVAL_FILE = (
    ROOT / 'dstarTL_results' / 'limo_T10_phi1_branch_intervals.csv'
)
OUTPUT_DIR = ROOT / 'figures'
TIME_HORIZON = 10
PLOT_STYLE = {
    'figsize': (7, 5),
    'font.size': 15,
    'axes.labelsize': 14,
    'axes.titlesize': 19,
    'legend.fontsize': 13,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'annotation.fontsize': 13,
}
SPEC_LABELS = {
    0: r'$\varphi_1$',
    1: r'$\varphi_1^{\prime}$',
    2: r'$\varphi_2$',
    3: r'$\varphi_2^{\prime}$',
    4: r'$\varphi_3$',
    5: r'$\varphi_4$',
}
SPEC_COLORS = {
    0: '#1f77b4',
    1: '#ff7f0e',
    2: '#2ca02c',
    3: '#d62728',
    4: '#9467bd',
    5: '#8c564b',
}


def load_limo_intervals(result_file, time_horizon):
    """Load one LIMO robustness interval per specification at the horizon."""
    intervals_by_spec = {}
    with result_file.open('r', encoding='utf-8') as result_stream:
        for line in result_stream:
            if not line.startswith('LIMO'):
                continue

            fields = line.split()
            if len(fields) not in (17, 18) or int(fields[1]) != time_horizon:
                continue

            spec_id = int(fields[2])
            classification_offset = 11 if len(fields) == 17 else 12
            intervals_by_spec[spec_id] = {
                'T': int(fields[1]),
                'spec_id': spec_id,
                'branches': int(fields[3]),
                'rho_lb': float(fields[4]),
                'rho_ub': float(fields[5]),
                'exact_lb': float(fields[6]),
                'exact_ub': float(fields[7]),
                'sat': int(fields[classification_offset]),
                'viol': int(fields[classification_offset + 1]),
                'mixed': int(fields[classification_offset + 2]),
            }

    return [intervals_by_spec[spec_id] for spec_id in sorted(intervals_by_spec)]


def save_interval_csv(intervals, output_file):
    """Save the LIMO values used in the robustness-interval figure."""
    fieldnames = [
        'T', 'spec_id', 'branches', 'rho_lb', 'rho_ub',
        'exact_lb', 'exact_ub',
        'sat', 'viol', 'mixed'
    ]
    with output_file.open('w', newline='', encoding='utf-8') as csv_stream:
        writer = csv.DictWriter(csv_stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(intervals)


def load_branch_intervals(branch_interval_file):
    """Load conservative and MILP intervals for each mixed phi1 branch."""
    intervals = []
    with branch_interval_file.open('r', encoding='utf-8') as csv_stream:
        for row in csv.DictReader(csv_stream):
            intervals.append({
                'T': int(row['T']),
                'spec_id': int(row['spec_id']),
                'branch_id': int(row['branch_id']),
                'rho_lb': float(row['rho_lb']),
                'rho_ub': float(row['rho_ub']),
                'exact_lb': float(row['exact_lb']),
                'exact_ub': float(row['exact_ub']),
            })

    return sorted(intervals, key=lambda interval: interval['branch_id'])


def add_interval(axis, y_position, lower_bound, upper_bound, color):
    """Draw one closed robustness interval."""
    axis.hlines(
        y_position,
        lower_bound,
        upper_bound,
        color=color,
        linewidth=5,
    )
    axis.plot(
        [lower_bound, upper_bound],
        [y_position, y_position],
        linestyle='None',
        marker='|',
        color=color,
        markersize=14,
        markeredgewidth=2.2,
    )


def plot_limo_intervals(intervals, output_dir):
    """Plot the LIMO robustness intervals for all specifications at T=10."""
    if not intervals:
        raise RuntimeError(
            'no LIMO rows found for T={} in {}'.format(
                TIME_HORIZON, RESULT_FILE
            )
        )

    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 11,
    })
    figure, axis = plt.subplots(figsize=(7.2, 4.1))
    row_positions = np.arange(len(intervals))[::-1]

    for row_position, interval in zip(row_positions, intervals):
        add_interval(
            axis,
            row_position,
            interval['rho_lb'],
            interval['rho_ub'],
            SPEC_COLORS[interval['spec_id']],
        )

    axis.axvline(
        0.0,
        color='black',
        linestyle='--',
        linewidth=1.3,
        label='Satisfaction boundary',
    )
    axis.set_yticks(row_positions)
    axis.set_yticklabels([
        SPEC_LABELS[interval['spec_id']] for interval in intervals
    ])
    for tick_label, interval in zip(axis.get_yticklabels(), intervals):
        tick_label.set_color(SPEC_COLORS[interval['spec_id']])

    axis.set_xlabel('Robustness value')
    # axis.set_title(r'LIMO Robustness Intervals at $T=10$')
    axis.set_ylim(-0.6, len(intervals) - 0.4)
    axis.grid(True, axis='x', linestyle=':', linewidth=0.7, alpha=0.7)
    axis.legend(loc='lower right', frameon=True)
    figure.tight_layout()

    output_stem = 'limo_T10_robustness_intervals'
    figure.savefig(
        output_dir / '{}.pdf'.format(output_stem),
        bbox_inches='tight',
    )
    figure.savefig(
        output_dir / '{}.png'.format(output_stem),
        dpi=300,
        bbox_inches='tight',
    )
    plt.close(figure)


def plot_limo_milp_refinement(intervals, output_dir):
    """Plot conservative and MILP-refined intervals for mixed cases."""
    mixed_intervals = [
        interval for interval in intervals
        if interval['mixed'] > 0
        and np.isfinite(interval['exact_lb'])
        and np.isfinite(interval['exact_ub'])
    ]
    if not mixed_intervals:
        raise RuntimeError(
            'no mixed LIMO rows with MILP intervals found for T={}'
            .format(TIME_HORIZON)
        )

    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 11,
    })
    figure_height = max(2.8, 1.25 * len(mixed_intervals) + 1.2)
    figure, axis = plt.subplots(figsize=(7.4, figure_height))

    group_positions = np.arange(len(mixed_intervals))[::-1]
    vertical_offset = 0.16
    conservative_color = '#7f7f7f'
    refined_color = '#ff7f0e'

    for group_position, interval in zip(group_positions, mixed_intervals):
        add_interval(
            axis,
            group_position + vertical_offset,
            interval['rho_lb'],
            interval['rho_ub'],
            conservative_color,
        )
        add_interval(
            axis,
            group_position - vertical_offset,
            interval['exact_lb'],
            interval['exact_ub'],
            refined_color,
        )
    axis.axvline(
        0.0,
        color='black',
        linestyle='--',
        linewidth=1.3,
        label=r'Satisfaction boundary ($\rho=0$)',
    )
    axis.set_yticks(group_positions)
    axis.set_yticklabels([
        '{}\n$N_{{\\mathrm{{mix}}}}={}$'.format(
            SPEC_LABELS[interval['spec_id']], interval['mixed']
        )
        for interval in mixed_intervals
    ])
    axis.set_xlabel('Robustness value')
    # axis.set_title(r'LIMO Mixed-Case Robustness Refinement at $T=10$')
    axis.set_ylim(-0.55, len(mixed_intervals) - 0.45)
    axis.grid(True, axis='x', linestyle=':', linewidth=0.7, alpha=0.7)
    axis.plot([], [], color=conservative_color, linewidth=5,
              label='Global conservative interval')
    axis.plot([], [], color=refined_color, linewidth=5,
              label='MILP-refined global interval')
    axis.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=True,
    )
    figure.tight_layout()

    output_stem = 'limo_T10_conservative_vs_milp_intervals'
    figure.savefig(
        output_dir / '{}.pdf'.format(output_stem),
        bbox_inches='tight',
    )
    figure.savefig(
        output_dir / '{}.png'.format(output_stem),
        dpi=300,
        bbox_inches='tight',
    )
    plt.close(figure)

    return mixed_intervals


def plot_limo_phi1_branch_refinement(branch_intervals, output_dir):
    """Plot two intervals for each mixed phi1 branch at T=10."""
    if not branch_intervals:
        raise RuntimeError(
            'no branch intervals found in {}'.format(BRANCH_INTERVAL_FILE)
        )
    if any(
            interval['T'] != TIME_HORIZON or interval['spec_id'] != 0
            for interval in branch_intervals):
        raise RuntimeError('branch interval file should contain only T=10 phi1')
    if any(
            not np.isfinite(interval[key])
            for interval in branch_intervals
            for key in ('rho_lb', 'rho_ub', 'exact_lb', 'exact_ub')):
        raise RuntimeError('every mixed branch requires two finite intervals')

    plt.rcParams.update({
        key: value for key, value in PLOT_STYLE.items()
        if key not in ('figsize', 'annotation.fontsize')
    })
    figure, axis = plt.subplots(figsize=PLOT_STYLE['figsize'])
    branch_positions = np.arange(1, len(branch_intervals) + 1)
    horizontal_offset = 0.14
    conservative_color = '#7f7f7f'
    refined_color = '#ff7f0e'

    for branch_position, interval in zip(branch_positions, branch_intervals):
        conservative_position = branch_position - horizontal_offset
        refined_position = branch_position + horizontal_offset
        axis.vlines(
            conservative_position,
            interval['rho_lb'],
            interval['rho_ub'],
            color=conservative_color,
            linewidth=5,
        )
        axis.plot(
            [conservative_position, conservative_position],
            [interval['rho_lb'], interval['rho_ub']],
            linestyle='None',
            marker='_',
            color=conservative_color,
            markersize=14,
            markeredgewidth=2.2,
        )
        axis.vlines(
            refined_position,
            interval['exact_lb'],
            interval['exact_ub'],
            color=refined_color,
            linewidth=5,
        )
        axis.plot(
            [refined_position, refined_position],
            [interval['exact_lb'], interval['exact_ub']],
            linestyle='None',
            marker='_',
            color=refined_color,
            markersize=14,
            markeredgewidth=2.2,
        )
        axis.text(
            refined_position + 0.07,
            0.5 * (interval['exact_lb'] + interval['exact_ub']),
            r'$[{:.6f},\ {:.6f}]$'.format(
                interval['exact_lb'], interval['exact_ub']
            ),
            color=refined_color,
            fontsize=PLOT_STYLE['annotation.fontsize'],
            rotation=90,
            ha='left',
            va='center',
        )

    axis.axhline(
        0.0,
        color='black',
        linestyle='--',
        linewidth=1.3,
    )
    axis.set_xticks(branch_positions)
    axis.set_xticklabels([
        str(interval['branch_id'] + 1)
        for interval in branch_intervals
    ])
    axis.set_xlabel('Branch')
    axis.set_ylabel('Robustness value')
    # axis.set_title(
    #     r'LIMO Branch Robustness Refinement for $\varphi_1$ at $T=10$'
    # )
    axis.set_xlim(0.5, len(branch_intervals) + 0.5)
    axis.grid(True, axis='y', linestyle=':', linewidth=0.7, alpha=0.7)
    figure.tight_layout()

    output_stem = 'limo_T10_phi1_branch_conservative_vs_milp_intervals'
    figure.savefig(
        output_dir / '{}.pdf'.format(output_stem),
        bbox_inches='tight',
    )
    figure.savefig(
        output_dir / '{}.png'.format(output_stem),
        dpi=300,
        bbox_inches='tight',
    )
    plt.close(figure)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    intervals = load_limo_intervals(RESULT_FILE, TIME_HORIZON)
    save_interval_csv(
        intervals,
        OUTPUT_DIR / 'limo_T10_robustness_intervals.csv',
    )
    plot_limo_intervals(intervals, OUTPUT_DIR)
    mixed_intervals = plot_limo_milp_refinement(intervals, OUTPUT_DIR)
    branch_intervals = load_branch_intervals(BRANCH_INTERVAL_FILE)
    plot_limo_phi1_branch_refinement(branch_intervals, OUTPUT_DIR)
    print(
        'Saved LIMO T={} interval figures for {} specifications; '
        '{} have mixed branches with MILP refinement; plotted {} phi1 branches'
        .format(
            TIME_HORIZON,
            len(intervals),
            len(mixed_intervals),
            len(branch_intervals),
        )
    )


if __name__ == '__main__':
    main()
