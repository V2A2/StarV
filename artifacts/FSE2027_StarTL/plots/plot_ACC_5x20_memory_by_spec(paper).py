"""Plot ACC 5x20 StarTL and ProbStarTL memory by specification."""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-starv-memory')

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUTS = [
    ROOT / 'memory_results' / 'acc_5x20_spec{}'.format(spec_id) / 'memory.csv'
    for spec_id in (0, 1, 2, 3, 4)
]
DEFAULT_OUTPUT_DIR = ROOT / 'figures'
METHODS = ('startl', 'probstartl')
METHOD_LABELS = {'startl': 'StarTL', 'probstartl': 'ProbStarTL'}
METHOD_COLORS = {'startl': '#1f77b4', 'probstartl': '#d62728'}
METHOD_MARKERS = {'startl': 'o', 'probstartl': 's'}
SPEC_LABELS = {
    0: r'$\varphi_1$',
    1: r"$\varphi_1^{\prime}$",
    2: r'$\varphi_2$',
    3: r"$\varphi_2^{\prime}$",
    4: r'$\varphi_3$',
}
METRIC_LABELS = {
    'peak_pss_gib': 'Peak PSS (GiB)',
    'peak_rss_gib': 'Peak RSS (GiB)',
    'incremental_pss_gib': 'Incremental memory usage (GiB)',
}
PLOT_STYLE = {
    'figsize': (7.5, 5),
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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Plot ACC 5x20 memory comparisons by specification.'
    )
    parser.add_argument('--inputs', nargs='+', type=Path, default=DEFAULT_INPUTS)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        '--metric',
        choices=list(METRIC_LABELS),
        default='peak_pss_gib',
    )
    return parser.parse_args()


def load_rows(input_files, metric):
    rows = []
    for input_file in input_files:
        if not input_file.exists():
            raise FileNotFoundError(input_file)
        with input_file.open('r', encoding='utf-8') as csv_stream:
            for row in csv.DictReader(csv_stream):
                if row['status'] != 'ok':
                    continue
                if row['system'] != 'ACC' or row['net'] != 'controller_5_20':
                    continue
                rows.append({
                    'spec_id': int(row['spec_id']),
                    'method': row['method'],
                    'horizon': int(row['horizon']),
                    'value': float(row[metric]),
                })
    if not rows:
        raise RuntimeError('no successful ACC controller_5_20 rows were found')
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            row['spec_id'], row['method'], row['horizon']
        ].append(row['value'])

    summaries = []
    for (spec_id, method, horizon), values in sorted(grouped.items()):
        values = np.asarray(values, dtype=float)
        summaries.append({
            'spec_id': spec_id,
            'method': method,
            'horizon': horizon,
            'median': float(np.median(values)),
            'minimum': float(np.min(values)),
            'maximum': float(np.max(values)),
            'repeats': int(values.size),
        })
    return summaries


def save_summary(summaries, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('w', newline='', encoding='utf-8') as csv_stream:
        writer = csv.DictWriter(csv_stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


def add_specification_panel(axis, summaries, spec_id):
    spec_rows = [row for row in summaries if row['spec_id'] == spec_id]
    for method in METHODS:
        method_rows = sorted(
            [row for row in spec_rows if row['method'] == method],
            key=lambda row: row['horizon'],
        )
        if not method_rows:
            continue
        horizons = np.asarray([row['horizon'] for row in method_rows])
        medians = np.asarray([row['median'] for row in method_rows])
        minima = np.asarray([row['minimum'] for row in method_rows])
        maxima = np.asarray([row['maximum'] for row in method_rows])
        axis.fill_between(
            horizons,
            minima,
            maxima,
            color=METHOD_COLORS[method],
            alpha=0.13,
            linewidth=0,
        )
        axis.plot(
            horizons,
            medians,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            markersize=5.5,
            linewidth=1.7,
            label=METHOD_LABELS[method],
        )

    # axis.set_title(SPEC_LABELS.get(spec_id, 'Spec. {}'.format(spec_id)))
    axis.set_xticks(sorted({row['horizon'] for row in spec_rows}))
    axis.set_ylim(bottom=0.0)
    axis.grid(True, linestyle=':', linewidth=0.65, alpha=0.65)


def plot_memory(summaries, metric, output_dir):
    apply_plot_style()
    spec_ids = sorted({row['spec_id'] for row in summaries})
    y_upper = 1.08 * max(row['maximum'] for row in summaries)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = []

    for spec_id in spec_ids:
        figure, axis = plt.subplots(figsize=PLOT_STYLE['figsize'])
        add_specification_panel(axis, summaries, spec_id)
        axis.set_xlabel('Time horizon, $T$')
        axis.set_ylabel(METRIC_LABELS[metric])
        axis.set_ylim(0.0, y_upper)
        axis.legend(frameon=False)
        figure.tight_layout()

        stem = 'acc_5x20_spec{}_{}_comparison'.format(spec_id, metric)
        png_file = output_dir / '{}.png'.format(stem)
        pdf_file = output_dir / '{}.pdf'.format(stem)
        figure.savefig(png_file, dpi=320, bbox_inches='tight')
        figure.savefig(pdf_file, bbox_inches='tight')
        plt.close(figure)
        output_files.append((png_file, pdf_file))

    return output_files


def main():
    arguments = parse_arguments()
    rows = load_rows(arguments.inputs, arguments.metric)
    summaries = summarize(rows)
    spec_ids = sorted({row['spec_id'] for row in summaries})
    if len(spec_ids) == 1:
        spec_tag = 'spec{}'.format(spec_ids[0])
    elif spec_ids == list(range(spec_ids[0], spec_ids[-1] + 1)):
        spec_tag = 'specs{}_{}'.format(spec_ids[0], spec_ids[-1])
    else:
        spec_tag = 'specs{}'.format('_'.join(map(str, spec_ids)))
    summary_file = (
        ROOT / 'memory_results'
        / 'acc_5x20_{}_memory_summary.csv'.format(spec_tag)
    )
    save_summary(summaries, summary_file)
    output_files = plot_memory(
        summaries,
        arguments.metric,
        arguments.output_dir,
    )
    print('Saved {}'.format(summary_file))
    for png_file, pdf_file in output_files:
        print('Saved {}'.format(png_file))
        print('Saved {}'.format(pdf_file))


if __name__ == '__main__':
    main()
