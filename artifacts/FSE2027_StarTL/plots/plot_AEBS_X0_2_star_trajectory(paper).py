"""Plot AEBS reachable Star trajectories for X0_2 at selected horizons."""

import contextlib
import csv
import io
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-starv')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Polygon, Rectangle
from scipy.spatial import ConvexHull

from StarV.nncs.nncs import AEBS_NNCS, ReachPRM_NNCS, reachDFS_DLNNCS
from StarV.util.load import load_AEBS_model_dStarTL
from StarV.util.plot import getVertices


ROOT = Path(__file__).resolve().parent
RESULT_FILE = ROOT / 'test_AEBS_exact_b130_T102030_full_specs_core1_new.txt'
OUTPUT_DIR = ROOT / 'figures'
HORIZONS = (10, 20, 30)
INITIAL_SET_ID = 2
INITIAL_SET_NAME = 'X0_{}'.format(INITIAL_SET_ID)
INITIAL_SET_LABEL = r'$X_0^2=[48,48.5]\times[30.2,30.4]$'
OUTPUT_STEM = 'aebs_X0_2_star_trajectories_T10_T20_T30'
DISTANCE_INDEX = 0
SPEED_INDEX = 1
UNSAFE_DISTANCE = 30.0
UNSAFE_SPEED = 0.2
PLOT_STYLE = {
    'figsize': (16.5, 6),
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


def load_reported_intervals(result_file):
    """Load X0_2 robustness intervals reported by the AEBS experiment."""
    intervals = {}
    with result_file.open('r', encoding='utf-8') as result_stream:
        for line in result_stream:
            if not line.startswith('AEBS'):
                continue
            fields = line.split()
            if len(fields) != 16 or fields[1] != INITIAL_SET_NAME:
                continue
            horizon = int(fields[2])
            if horizon in HORIZONS:
                intervals[horizon] = {
                    'rho_lb': float(fields[5]),
                    'rho_ub': float(fields[6]),
                    'branches': int(fields[4]),
                    'sat_branches': int(fields[10]),
                    'viol_branches': int(fields[11]),
                    'mixed_branches': int(fields[12]),
                }

    missing = sorted(set(HORIZONS) - set(intervals))
    if missing:
        raise RuntimeError(
            'missing {} result rows for T={}'.format(INITIAL_SET_NAME, missing)
        )
    return intervals


def compute_reachable_traces(horizon):
    """Run exact AEBS reachability for X0_2 and one horizon."""
    controller, transformer, norm_mat, scale_mat, plant, initial_sets = (
        load_AEBS_model_dStarTL()
    )
    aebs = AEBS_NNCS(controller, transformer, norm_mat, scale_mat, plant)
    reach_parameters = ReachPRM_NNCS()
    reach_parameters.initSet = initial_sets[INITIAL_SET_ID]
    reach_parameters.numSteps = horizon
    reach_parameters.filterProb = 0.0
    reach_parameters.lpSolver = 'linprog'
    reach_parameters.show = False
    reach_parameters.numCores = 1

    quiet_output = io.StringIO()
    with contextlib.redirect_stdout(quiet_output):
        traces, _ = reachDFS_DLNNCS(aebs, reach_parameters)
    return traces


def project_star_vertices(star):
    """Return ordered vertices of a Star projected onto distance and speed."""
    projection = np.zeros((2, star.dim))
    projection[0, DISTANCE_INDEX] = 1.0
    projection[1, SPEED_INDEX] = 1.0
    projected_star = star.affineMap(projection)
    vertices = np.asarray(getVertices(projected_star), dtype=float)
    if vertices.ndim != 2 or vertices.shape[0] == 0:
        return np.empty((0, 2))
    if vertices.shape[0] > 2:
        vertices = vertices[ConvexHull(vertices).vertices]
    return vertices


def prepare_trace_geometry(traces):
    """Project every Star once and collect representative branch trajectories."""
    vertex_cache = {}
    branch_paths = []
    all_vertices = []

    for trace in traces:
        branch_path = []
        for star in trace:
            cache_key = id(star)
            vertices = vertex_cache.get(cache_key)
            if vertices is None:
                vertices = project_star_vertices(star)
                vertex_cache[cache_key] = vertices
            if vertices.size == 0:
                continue
            branch_path.append(vertices.mean(axis=0))
            all_vertices.append(vertices)
        if branch_path:
            branch_paths.append(np.asarray(branch_path))

    return vertex_cache, branch_paths, all_vertices


def add_reachable_stars(axis, traces, horizon, colormap, time_norm):
    """Draw projected Star polygons and branch-center trajectories."""
    plotted_keys = set()
    branch_paths = []
    all_vertices = []

    for trace in traces:
        path = []
        for time_index, star in enumerate(trace):
            vertices = project_star_vertices(star)
            if vertices.size == 0:
                continue
            path.append(vertices.mean(axis=0))
            all_vertices.append(vertices)

            polygon_key = (
                time_index,
                tuple(np.round(vertices.ravel(), decimals=8)),
            )
            if polygon_key in plotted_keys:
                continue
            plotted_keys.add(polygon_key)
            color = colormap(time_norm(time_index))
            if vertices.shape[0] >= 3:
                axis.add_patch(Polygon(
                    vertices,
                    closed=True,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0.45,
                    alpha=0.20,
                ))
            else:
                axis.plot(
                    vertices[:, 0], vertices[:, 1],
                    color=color, linewidth=0.8, alpha=0.55,
                )

        if path:
            branch_paths.append(np.asarray(path))

    for path in branch_paths:
        times = np.arange(path.shape[0])
        axis.scatter(
            path[:, 0], path[:, 1],
            c=times, cmap=colormap, norm=time_norm,
            s=4.5, alpha=0.65, linewidths=0,
        )
        axis.plot(
            path[:, 0], path[:, 1],
            color='#2b8c3e', linestyle=':', linewidth=0.65, alpha=0.40,
        )

    if not all_vertices:
        raise RuntimeError('no reachable Star projection was generated for T={}'.format(horizon))
    return np.vstack(all_vertices)


def plot_aebs_trajectory(traces_by_horizon, intervals, output_dir):
    """Create one AEBS X0_2 trajectory panel for each selected horizon."""
    apply_plot_style()
    figure, axes = plt.subplots(
        1,
        3,
        figsize=PLOT_STYLE['figsize'],
        sharex=True,
        sharey=True,
    )
    colormap = colormaps.get_cmap('YlGn')

    projected_vertices = {}
    for horizon, axis in zip(HORIZONS, axes):
        time_norm = Normalize(vmin=0, vmax=horizon)
        projected_vertices[horizon] = add_reachable_stars(
            axis,
            traces_by_horizon[horizon],
            horizon,
            colormap,
            time_norm,
        )

    combined_vertices = np.vstack(list(projected_vertices.values()))
    distance_max = max(50.0, float(np.max(combined_vertices[:, 0])) + 2.0)
    speed_max = max(26.0, float(np.max(combined_vertices[:, 1])) + 1.0)

    for horizon, axis in zip(HORIZONS, axes):
        axis.add_patch(Rectangle(
            (0.0, UNSAFE_SPEED),
            UNSAFE_DISTANCE,
            speed_max - UNSAFE_SPEED,
            facecolor='#ef5350',
            edgecolor='#c62828',
            linewidth=1.0,
            alpha=0.18,
            zorder=-5,
        ))
        axis.axvline(
            UNSAFE_DISTANCE,
            color='#c62828', linestyle='--', linewidth=1.15,
        )
        axis.axhline(
            UNSAFE_SPEED,
            xmax=UNSAFE_DISTANCE / distance_max,
            color='#c62828', linestyle='--', linewidth=1.0,
        )
        interval = intervals[horizon]
        axis.set_title(
            (
                r'$T={},\quad '
                r'(N_{{\mathrm{{sat}}}},\,N_{{\mathrm{{viol}}}},\,'
                r'N_{{\mathrm{{mix}}}})=({},{},{})$'
            ).format(
                horizon,
                interval['sat_branches'],
                interval['viol_branches'],
                interval['mixed_branches'],
            ),
            pad=2,
)
        
        axis.set_xlim(0.0, distance_max)
        axis.set_ylim(-1.2, speed_max)
        axis.set_xlabel(r'Distance $d$')
        axis.grid(True, linestyle=':', linewidth=0.55, alpha=0.55)

    speed_ticks = np.concatenate((
        np.array([UNSAFE_SPEED]),
        np.arange(5.0, speed_max + 0.1, 5.0),
    ))
    axes[0].set_yticks(speed_ticks)
    axes[0].set_yticklabels([
        '0.2' if np.isclose(tick, UNSAFE_SPEED) else '{:g}'.format(tick)
        for tick in speed_ticks
    ])
    axes[0].set_ylabel(r'Ego speed $v_{\mathrm{ego}}$')
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 0.99))

    figure.savefig(output_dir / '{}.png'.format(OUTPUT_STEM), dpi=350, bbox_inches='tight')
    figure.savefig(output_dir / '{}.pdf'.format(OUTPUT_STEM), bbox_inches='tight')
    plt.close(figure)


def save_summary(intervals, traces_by_horizon, output_file):
    """Save the values displayed in the figure."""
    with output_file.open('w', newline='', encoding='utf-8') as csv_stream:
        writer = csv.DictWriter(
            csv_stream,
            fieldnames=['T', 'branches', 'computed_traces', 'rho_lb', 'rho_ub'],
        )
        writer.writeheader()
        for horizon in HORIZONS:
            writer.writerow({
                'T': horizon,
                'branches': intervals[horizon]['branches'],
                'computed_traces': len(traces_by_horizon[horizon]),
                'rho_lb': intervals[horizon]['rho_lb'],
                'rho_ub': intervals[horizon]['rho_ub'],
            })


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    intervals = load_reported_intervals(RESULT_FILE)
    traces_by_horizon = {}
    for horizon in HORIZONS:
        print('Computing AEBS {} reachable traces for T={}...'.format(
            INITIAL_SET_NAME, horizon
        ))
        traces_by_horizon[horizon] = compute_reachable_traces(horizon)
        expected = intervals[horizon]['branches']
        actual = len(traces_by_horizon[horizon])
        if actual != expected:
            raise RuntimeError(
                'T={} generated {} traces, but the result table reports {}'
                .format(horizon, actual, expected)
            )

    save_summary(
        intervals,
        traces_by_horizon,
        OUTPUT_DIR / '{}.csv'.format(OUTPUT_STEM),
    )
    plot_aebs_trajectory(traces_by_horizon, intervals, OUTPUT_DIR)
    print('Saved AEBS X0_2 reachable Star trajectory figure')


if __name__ == '__main__':
    main()
