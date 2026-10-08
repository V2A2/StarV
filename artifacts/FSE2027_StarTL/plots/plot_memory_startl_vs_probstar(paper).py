"""Plot StarTL versus ProbStarTL peak process-tree memory."""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-starv-memory')

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / 'memory_results' / 'startl_vs_probstar_memory.csv'
DEFAULT_OUTPUT_DIR = ROOT / 'figures'
METHOD_LABELS = {'startl': 'StarTL', 'probstartl': 'ProbStarTL'}
METHOD_COLORS = {'startl': '#1f77b4', 'probstartl': '#d62728'}
METHOD_MARKERS = {'startl': 'o', 'probstartl': 's'}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        '--metric',
        choices=['peak_pss_gib', 'peak_rss_gib', 'incremental_pss_gib'],
        default='peak_pss_gib',
    )
    return parser.parse_args()


def load_rows(input_file):
    rows = []
    with input_file.open('r', encoding='utf-8') as csv_stream:
        for row in csv.DictReader(csv_stream):
            if row['status'] != 'ok':
                continue
            row['horizon'] = int(row['horizon'])
            row['init_set'] = (
                int(row['init_set']) if row['init_set'] != '' else None
            )
            row['branches'] = (
                int(float(row['branches'])) if row['branches'] != '' else None
            )
            for field in (
                    'peak_pss_gib', 'peak_rss_gib', 'incremental_pss_gib'):
                row[field] = float(row[field])
            rows.append(row)
    if not rows:
        raise RuntimeError('no successful memory rows found in {}'.format(input_file))
    return rows


def summarize(rows, metric):
    grouped = defaultdict(list)
    branch_counts = defaultdict(list)
    for row in rows:
        configuration = (
            row['net'] if row['system'] == 'ACC'
            else 'X0_{}'.format(row['init_set'])
        )
        key = (
            row['system'], configuration, row['method'], row['horizon']
        )
        grouped[key].append(row[metric])
        if row['branches'] is not None:
            branch_counts[key].append(row['branches'])

    summaries = []
    for key, values in grouped.items():
        values = np.asarray(values, dtype=float)
        branches = branch_counts[key]
        summaries.append({
            'system': key[0],
            'configuration': key[1],
            'method': key[2],
            'horizon': key[3],
            'median': float(np.median(values)),
            'minimum': float(np.min(values)),
            'maximum': float(np.max(values)),
            'repeats': int(values.size),
            'branches': int(np.median(branches)) if branches else '',
        })
    return summaries


def save_summary(summaries, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('w', newline='', encoding='utf-8') as csv_stream:
        writer = csv.DictWriter(csv_stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(sorted(
            summaries,
            key=lambda row: (
                row['system'], row['configuration'],
                row['method'], row['horizon']
            ),
        ))


def add_configuration_panel(axis, summaries, system, configuration, metric):
    panel_rows = [
        row for row in summaries
        if row['system'] == system and row['configuration'] == configuration
    ]
    for method in ('startl', 'probstartl'):
        method_rows = sorted(
            [row for row in panel_rows if row['method'] == method],
            key=lambda row: row['horizon'],
        )
        if not method_rows:
            continue
        horizons = np.asarray([row['horizon'] for row in method_rows])
        medians = np.asarray([row['median'] for row in method_rows])
        lower = medians - np.asarray([row['minimum'] for row in method_rows])
        upper = np.asarray([row['maximum'] for row in method_rows]) - medians
        axis.errorbar(
            horizons,
            medians,
            yerr=np.vstack((lower, upper)),
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            markersize=5.5,
            linewidth=1.6,
            capsize=3,
            label=METHOD_LABELS[method],
        )

    # axis.set_title(
    #     configuration.replace('controller_', r'$N_{').replace('_', r'\times')
    #     + '}$' if system == 'ACC' else configuration.replace('_', r'$_') + '$'
    # )
    axis.set_xlabel('Time horizon, $T$')
    axis.set_ylabel(
        'Memory usage (GiB)'
        if metric == 'peak_pss_gib'
        else metric.replace('_', ' ')
    )
    axis.grid(True, linestyle=':', linewidth=0.65, alpha=0.65)
    axis.legend(fontsize=8.5)


def plot_system(summaries, system, metric, output_dir):
    configurations = sorted({
        row['configuration'] for row in summaries if row['system'] == system
    })
    if not configurations:
        return None, None

    if system == 'AEBS' and len(configurations) > 2:
        rows, columns = 2, int(np.ceil(len(configurations) / 2.0))
    else:
        rows, columns = 1, len(configurations)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.2 * columns, 3.4 * rows),
        squeeze=False,
        sharey=True,
    )
    flat_axes = axes.ravel()
    for axis, configuration in zip(flat_axes, configurations):
        add_configuration_panel(axis, summaries, system, configuration, metric)
    for axis in flat_axes[len(configurations):]:
        axis.set_visible(False)

    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = '{}_startl_vs_probstar_peak_memory'.format(system.lower())
    png_file = output_dir / '{}.png'.format(stem)
    pdf_file = output_dir / '{}.pdf'.format(stem)
    figure.savefig(png_file, dpi=320, bbox_inches='tight')
    figure.savefig(pdf_file, bbox_inches='tight')
    plt.close(figure)
    return png_file, pdf_file


def main():
    arguments = parse_arguments()
    rows = load_rows(arguments.input)
    summaries = summarize(rows, arguments.metric)
    summary_file = (
        arguments.input.parent / 'startl_vs_probstar_memory_summary.csv'
    )
    save_summary(summaries, summary_file)
    print('Saved {}'.format(summary_file))

    for system in ('ACC', 'AEBS'):
        png_file, pdf_file = plot_system(
            summaries, system, arguments.metric, arguments.output_dir
        )
        if png_file is not None:
            print('Saved {}'.format(png_file))
            print('Saved {}'.format(pdf_file))


if __name__ == '__main__':
    main()
