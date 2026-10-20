import csv
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-starv')

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULT_FILES = {
    '3x20': (
        ROOT / 'dstarTL_results'
        / 'test_acc_exact_3_20NN_T1020304050_full_sepcs_core1.txt'
    ),
    '5x20': (
        ROOT / 'dstarTL_results'
        / 'test_acc_exact_5_20NN_T1020304050_full_sepcs_core1.txt'
    ),
}
OUTPUT_DIR = ROOT / 'figures'
HORIZONS = (10, 20, 30, 40,50)
BRANCH_HORIZONS = (10, 20, 30, 40, 50)
SPEC_LABELS = {
    0: r'$\varphi_1$',
    1: r'$\varphi_1^{\prime}$',
    2: r'$\varphi_2$',
    3: r'$\varphi_2^{\prime}$',
    4: r'$\varphi_3$',
    5: r'$\varphi_4$',
}
SPEC_FILENAMES = {
    0: 'phi1',
    1: 'phi1_prime',
    2: 'phi2',
    3: 'phi2_prime',
    4: 'phi3',
    5: 'phi4',
}
PLOT_STYLE = {
    'figsize': (7.5, 5.2),
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


def load_runtime_data(result_file):
    """Load reported ACC runtimes by specification and horizon."""
    rows_by_spec = {spec_id: [] for spec_id in SPEC_LABELS}
    seen_rows = set()

    with result_file.open('r', encoding='utf-8') as result_stream:
        for line in result_stream:
            if not line.startswith('ACC'):
                continue

            fields = line.split()
            if len(fields) != 15:
                continue

            horizon = int(fields[1])
            spec_id = int(fields[2])
            row_key = (spec_id, horizon)
            if (
                    horizon not in HORIZONS
                    or spec_id not in SPEC_LABELS
                    or row_key in seen_rows):
                continue

            rows_by_spec[spec_id].append({
                'spec_id': spec_id,
                'T': horizon,
                'reach_time': float(fields[12]),
                'checking_time': float(fields[13]),
                'verification_time': float(fields[14]),
            })
            seen_rows.add(row_key)

    for spec_id in rows_by_spec:
        rows_by_spec[spec_id].sort(key=lambda row: row['T'])
        available_horizons = tuple(row['T'] for row in rows_by_spec[spec_id])
        if available_horizons != HORIZONS:
            raise RuntimeError(
                'expected horizons {} for specification {}, found {}'
                .format(HORIZONS, spec_id, available_horizons)
            )

    return rows_by_spec


def save_runtime_csv(rows_by_spec, output_file):
    """Save all values used to generate the per-specification figures."""
    with output_file.open('w', newline='', encoding='utf-8') as csv_stream:
        writer = csv.DictWriter(
            csv_stream,
            fieldnames=[
                'spec_id', 'T', 'reach_time', 'checking_time',
                'verification_time'
            ]
        )
        writer.writeheader()
        for spec_id in sorted(rows_by_spec):
            writer.writerows(rows_by_spec[spec_id])



def load_branch_data(result_file):
    """Load one branch count for every requested horizon."""
    branches_by_horizon = {}
    with result_file.open('r', encoding='utf-8') as result_stream:
        for line in result_stream:
            if not line.startswith('ACC'):
                continue

            fields = line.split()
            if len(fields) != 15:
                continue

            horizon = int(fields[1])
            if horizon in BRANCH_HORIZONS and horizon not in branches_by_horizon:
                branches_by_horizon[horizon] = int(fields[3])

    available_horizons = tuple(sorted(branches_by_horizon))
    if available_horizons != BRANCH_HORIZONS:
        raise RuntimeError(
            'expected branch horizons {}, found {}'
            .format(BRANCH_HORIZONS, available_horizons)
        )

    return [
        {'T': horizon, 'branches': branches_by_horizon[horizon]}
        for horizon in BRANCH_HORIZONS
    ]


def save_branch_csv(all_branch_data, output_file):
    """Save branch counts for both ACC controllers."""
    with output_file.open('w', newline='', encoding='utf-8') as csv_stream:
        writer = csv.DictWriter(
            csv_stream,
            fieldnames=['network', 'T', 'branches']
        )
        writer.writeheader()
        for network, branch_data in all_branch_data.items():
            for row in branch_data:
                writer.writerow({'network': network, **row})

def plot_specification_runtime(network, spec_id, runtime_data, output_dir):
    """Plot reachability, checking, and total runtime for one specification."""
    phase_styles = {
        'reach_time': ('Reachability $t_r$', '#1f77b4', 'o'),
        'checking_time': ('Specification checking $t_c$', '#d95f02', 's'),
        'verification_time': ('Total verification $t_v$', '#1b9e77', '^'),
    }

    apply_plot_style()
    figure, axis = plt.subplots(figsize=PLOT_STYLE['figsize'])
    horizons = [row['T'] for row in runtime_data]

    for metric, (phase_name, color, marker) in phase_styles.items():
        axis.plot(
            horizons,
            [row[metric] for row in runtime_data],
            label=phase_name,
            color=color,
            marker=marker,
            linewidth=2.3,
            markersize=8,
        )

    axis.set_xlabel('Time horizon, $T$')
    axis.set_ylabel('Runtime (seconds)')
    # axis.set_title(
    #     r'ACC Net $NETWORK$: {} Runtime'.format(SPEC_LABELS[spec_id])
    #     .replace('NETWORK', network.replace('x', r'\times'))
    # )
    axis.set_yscale('log')
    axis.set_xticks(HORIZONS)
    axis.grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.7)
    axis.legend(frameon=True, loc='upper left')
    figure.tight_layout()

    output_stem = 'acc_{}_{}_runtime_vs_horizon'.format(
        network, SPEC_FILENAMES[spec_id]
    )
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



def plot_branch_growth(all_branch_data, output_dir):
    """Plot grouped ACC branch counts for the 3x20 and 5x20 controllers."""
    apply_plot_style()
    network_styles = {
        '3x20': (r'Net$_{3\times20}$', '#1f77b4'),
        '5x20': (r'Net$_{5\times20}$', '#d95f02'),
    }
    x_positions = list(range(len(BRANCH_HORIZONS)))
    bar_width = 0.36
    network_offsets = {
        '3x20': -bar_width / 2.0,
        '5x20': bar_width / 2.0,
    }

    figure, axis = plt.subplots(figsize=PLOT_STYLE['figsize'])
    for network, branch_data in all_branch_data.items():
        label, color = network_styles[network]
        branch_counts = [row['branches'] for row in branch_data]
        bar_positions = [
            position + network_offsets[network] for position in x_positions
        ]
        bars = axis.bar(
            bar_positions,
            branch_counts,
            width=bar_width,
            label=label,
            color=color,
            edgecolor='black',
            linewidth=0.6,
        )
        axis.bar_label(
            bars,
            labels=[str(branch_count) for branch_count in branch_counts],
            padding=3,
            color=color,
        )

    axis.set_xlabel('Time horizon, $T$')
    axis.set_ylabel('Number of ReLU branches')
    # axis.set_title('ACC Branch Growth')
    axis.set_xticks(x_positions)
    axis.set_xticklabels(BRANCH_HORIZONS)
    axis.set_ylim(bottom=0)
    axis.grid(True, axis='y', linestyle=':', linewidth=0.7, alpha=0.7)
    axis.legend(frameon=True, loc='upper left')
    figure.tight_layout()

    output_stem = 'acc_3x20_5x20_branch_count_vs_horizon'
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
    all_branch_data = {}

    for network, result_file in RESULT_FILES.items():
        rows_by_spec = load_runtime_data(result_file)
        all_branch_data[network] = load_branch_data(result_file)
        save_runtime_csv(
            rows_by_spec,
            OUTPUT_DIR / 'acc_{}_runtime_vs_horizon.csv'.format(network),
        )

        for spec_id, runtime_data in rows_by_spec.items():
            plot_specification_runtime(
                network, spec_id, runtime_data, OUTPUT_DIR
            )
            print(
                'Saved ACC Net {} runtime figure for specification {}'
                .format(network, spec_id)
            )

    save_branch_csv(
        all_branch_data,
        OUTPUT_DIR / 'acc_3x20_5x20_branch_count_vs_horizon.csv',
    )
    plot_branch_growth(all_branch_data, OUTPUT_DIR)
    print('Saved combined ACC branch-growth figure')


if __name__ == '__main__':
    main()
