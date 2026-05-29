"""
2D Star reachability tutorial using StarV native APIs.

This script gives a small visual example of what reachability means in
verification:

1. Create a 2D input Star set from lower and upper state bounds.
2. Draw concrete samples from that Star set.
3. Propagate the Star through a tiny ReLU network with exact reachability.
   The ReLU layer can split one input Star into multiple exact reachable Stars.
4. Plot exact output reachability, over-approximate output reachability, and a combined comparison figure.

The figures are saved next to this file in ./figures/.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from StarV.set.star import Star
from StarV.layer.ReLULayer import ReLULayer
from StarV.layer.FullyConnectedLayer import FullyConnectedLayer
from StarV.net.network import NeuralNetwork, reachApproxBFS
from StarV.util.lp_solver import sample
from StarV.util.plot import plot_star


EXACT_STAR_COLORS = [
    "#f4a261",
    "#2a9d8f",
    "#e76f51",
    "#1766DC",
    "#8ab17d",
    "#b56576",
]
OVERAPPROX_COLOR = "#C91818"
SAMPLE_COLOR = "#001219"
INPUT_COLOR = "#8ecae6"


def create_tiny_relu_network() -> NeuralNetwork:
    """Create a 2D -> ReLU -> 2D network for the tutorial."""
    W1 = np.array([[1.0, -0.8], [-0.6, 1.1]])
    b1 = np.array([0.15, -0.05])
    W2 = np.array([[1.0, 0.55], [-0.65, 1.0]])
    b2 = np.array([-0.05, 0.1])

    layers = [
        FullyConnectedLayer([W1, b1]),
        ReLULayer(),
        FullyConnectedLayer([W2, b2]),
    ]
    return NeuralNetwork(layers, net_type="ffnn_2d_star_reachability_demo")


def reach_exact_with_hidden_sets(network: NeuralNetwork, input_star: Star, lp_solver: str = "linprog") -> tuple[list[Star], list[Star]]:
    """Run exact reachability and keep the Stars immediately after ReLU."""
    reachable_sets = [input_star]
    hidden_exact_sets = None

    for layer in network.layers:
        reachable_sets = layer.reach(reachable_sets, method="exact", lp_solver=lp_solver, pool=None)
        if isinstance(layer, ReLULayer):
            hidden_exact_sets = reachable_sets

    if hidden_exact_sets is None:
        hidden_exact_sets = reachable_sets

    return hidden_exact_sets, reachable_sets


def reach_approx_with_hidden_set(network: NeuralNetwork, input_star: Star, lp_solver: str = "linprog") -> Star:
    """Run over-approximate reachability up to the ReLU layer."""
    reachable_set = input_star

    for layer in network.layers:
        reachable_set = layer.reach(reachable_set, method="approx", lp_solver=lp_solver, pool=None)
        if isinstance(layer, ReLULayer):
            return reachable_set

    return reachable_set


def as_list(reach_set):
    """Normalize StarV reachability output to a list for plotting/counting."""
    return reach_set if isinstance(reach_set, list) else [reach_set]


def add_plot_labels(ax: plt.Axes, labels: list[tuple[str, str]]) -> None:
    handles = [mpatches.Patch(color=color, label=label) for label, color in labels]
    ax.legend(handles=handles, fontsize=8, loc="lower left")


def add_reachability_legend(
    ax: plt.Axes,
    *,
    overapprox_label: str | None = None,
    exact_labels: list[tuple[str, str]],
    sample_label: str | None = None,
) -> None:
    handles = []
    if overapprox_label is not None:
        handles.append(mpatches.Patch(color=OVERAPPROX_COLOR, label=overapprox_label))
    handles.extend(mpatches.Patch(color=color, label=label) for label, color in exact_labels)
    if sample_label is not None:
        handles.append(
            mlines.Line2D([], [], color=SAMPLE_COLOR, marker="o", linestyle="None", markersize=5, label=sample_label)
        )
    ax.legend(handles=handles, fontsize=8, loc="lower left")


def setup_reachability_axis(ax: plt.Axes) -> None:
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.set_aspect("equal", adjustable="box")


def star_bounds(reach_sets, lp_solver: str = "linprog") -> tuple[np.ndarray, np.ndarray]:
    """Return coordinate-wise bounds for one Star or a list of Stars."""
    stars = as_list(reach_sets)
    lows = []
    highs = []
    for reach_set in stars:
        lb, ub = reach_set.getRanges(lp_solver=lp_solver)
        lows.append(lb)
        highs.append(ub)
    return np.vstack(lows).min(axis=0), np.vstack(highs).max(axis=0)


def sample_bounds(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return coordinate-wise bounds for 2D sample columns."""
    return samples.min(axis=1), samples.max(axis=1)


def apply_plot_bounds(ax: plt.Axes, *bounds: tuple[np.ndarray, np.ndarray], pad_ratio: float = 0.08) -> None:
    """Apply common padded bounds after StarV plotting resets axis limits."""
    lows = np.vstack([lb for lb, _ in bounds])
    highs = np.vstack([ub for _, ub in bounds])
    lb = lows.min(axis=0)
    ub = highs.max(axis=0)
    span = ub - lb
    pad = np.maximum(span * pad_ratio, 0.08)
    ax.set_xlim(lb[0] - pad[0], ub[0] + pad[0])
    ax.set_ylim(lb[1] - pad[1], ub[1] + pad[1])


def make_axis_artists_opaque(
    ax: plt.Axes,
    start_patch_count: int,
    start_collection_count: int,
    color: str,
    zorder: float,
) -> None:
    for patch in ax.patches[start_patch_count:]:
        patch.set_alpha(1.0)
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_zorder(zorder)
    for collection in ax.collections[start_collection_count:]:
        collection.set_alpha(1.0)
        collection.set_facecolor(color)
        collection.set_edgecolor(color)
        collection.set_zorder(zorder)


def plot_starv_set(
    ax: plt.Axes,
    reach_set,
    *,
    color: str,
    zorder: float = 2.0,
    lp_solver: str = "linprog",
) -> None:
    for star in as_list(reach_set):
        lb, ub = star.getRanges(lp_solver=lp_solver)
        span = ub - lb
        if np.allclose(span, 0.0, atol=1e-9):
            ax.scatter([lb[0]], [lb[1]], s=70, color=color, edgecolor=color, zorder=zorder)
        elif np.isclose(span[0], 0.0, atol=1e-9):
            ax.plot(
                [lb[0], lb[0]],
                [lb[1], ub[1]],
                color=color,
                linewidth=3.2,
                solid_capstyle="round",
                zorder=zorder,
            )
        elif np.isclose(span[1], 0.0, atol=1e-9):
            ax.plot(
                [lb[0], ub[0]],
                [lb[1], lb[1]],
                color=color,
                linewidth=3.2,
                solid_capstyle="round",
                zorder=zorder,
            )
        else:
            patch_count = len(ax.patches)
            collection_count = len(ax.collections)
            plt.sca(ax)
            plot_star(star, show=False, color=color)
            make_axis_artists_opaque(ax, patch_count, collection_count, color, zorder)


def plot_numbered_exact_stars(
    ax: plt.Axes,
    reach_sets: list[Star],
    *,
    label_prefix: str,
    color: str | list[str] | tuple[str, ...] | None = None,
    zorder: float = 2.0,
) -> list[tuple[str, str]]:
    labels = []
    for i, reach_set in enumerate(reach_sets, start=1):
        if isinstance(color, (list, tuple)):
            star_color = color[(i - 1) % len(color)]
        elif color is not None:
            star_color = color
        else:
            star_color = EXACT_STAR_COLORS[(i - 1) % len(EXACT_STAR_COLORS)]
        plot_starv_set(ax, reach_set, color=star_color, zorder=zorder)
        labels.append((f"{label_prefix}{i}", star_color))
    return labels


def main() -> None:
    # Fix the random seed so the sampled points in the tutorial figure are
    # reproducible each time the script is run.
    np.random.seed(7)

    # Build a tiny 2D feedforward neural network:
    # input -> fully connected -> ReLU -> fully connected -> output.
    network = create_tiny_relu_network()

    # Define the uncertain 2D input region using lower and upper bounds.
    # Star(input_lb, input_ub) creates a box-shaped Star set:
    #     input_lb <= x <= input_ub.
    input_lb = np.array([-1.0, -0.8])
    input_ub = np.array([1.0, 0.9])
    input_star = Star(input_lb, input_ub)

    # Draw concrete input samples from the Star set and evaluate the network
    # on those points. These samples are only for visualization; reachability
    # is computed symbolically with Star sets below.
    input_samples = sample(input_star, 1000, lp_solver="linprog")
    hidden_samples = network.layers[1].evaluate(network.layers[0].evaluate(input_samples))
    output_samples = network.evaluate(input_samples)

    # Exact reachability propagates the input Star through each layer. At the
    # ReLU layer, StarV splits the set whenever a neuron can be both active and
    # inactive, producing multiple exact reachable Stars.
    hidden_exact_sets, output_exact_sets = reach_exact_with_hidden_sets(network, input_star, lp_solver="linprog")

    # Approximate ReLU reachability keeps one Star enclosing the exact hidden
    # Stars. Output approximate reachability is computed with StarV reachApproxBFS.
    hidden_overapprox_set = reach_approx_with_hidden_set(network, input_star, lp_solver="linprog")
    output_overapprox_sets = as_list(reachApproxBFS(network, input_star, lp_solver="linprog", show=False))

    # Compute common plot bounds. StarV plot_star resets limits for each
    # individual Star, so these bounds are applied after drawing to keep
    # lower-dimensional exact Stars from being visually cut off.
    input_plot_bounds = (star_bounds([input_star]), sample_bounds(input_samples))
    hidden_plot_bounds = (
        star_bounds(hidden_exact_sets),
        star_bounds([hidden_overapprox_set]),
        sample_bounds(hidden_samples),
    )
    output_plot_bounds = (
        star_bounds(output_exact_sets),
        star_bounds(output_overapprox_sets),
        sample_bounds(output_samples),
    )

    # Compute output intervals for printing.
    exact_output_lb, exact_output_ub = star_bounds(output_exact_sets)
    overapprox_lb, overapprox_ub = output_overapprox_sets[0].getRanges(lp_solver="linprog")

    # Save the tutorial figures next to the script. The exact and
    # over-approximate output sets are plotted separately so their shapes are
    # easy to compare without one covering the other.
    figure_dir = Path(__file__).resolve().parent / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: input, exact hidden ReLU Stars, and exact output Stars.
    fig_exact, axes_exact = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    fig_exact.suptitle("Exact Star Reachability for a 2D ReLU Network", fontsize=16, fontweight="bold")

    setup_reachability_axis(axes_exact[0])
    plot_starv_set(axes_exact[0], input_star, color=INPUT_COLOR)
    axes_exact[0].scatter(input_samples[0, :], input_samples[1, :], s=10, color="#023047", zorder=5)
    axes_exact[0].set_xlabel(r"$x_1$")
    axes_exact[0].set_ylabel(r"$x_2$")
    axes_exact[0].set_title("Input Set")
    add_plot_labels(axes_exact[0], [("Input Star", INPUT_COLOR), ("Samples", "#023047")])
    apply_plot_bounds(axes_exact[0], *input_plot_bounds)

    setup_reachability_axis(axes_exact[1])
    hidden_exact_labels = plot_numbered_exact_stars(
        axes_exact[1], hidden_exact_sets, label_prefix="Exact ReLU Star"
    )
    axes_exact[1].scatter(hidden_samples[0, :], hidden_samples[1, :], s=10, color=SAMPLE_COLOR, zorder=5)
    axes_exact[1].set_xlabel(r"$h_1$")
    axes_exact[1].set_ylabel(r"$h_2$")
    axes_exact[1].set_title("Exact Intermediate Sets")
    add_reachability_legend(
        axes_exact[1],
        exact_labels=hidden_exact_labels,
        sample_label="ReLU samples",
    )
    apply_plot_bounds(axes_exact[1], *hidden_plot_bounds)

    setup_reachability_axis(axes_exact[2])
    output_exact_labels = plot_numbered_exact_stars(
        axes_exact[2], output_exact_sets, label_prefix="Exact output Star"
    )
    axes_exact[2].scatter(output_samples[0, :], output_samples[1, :], s=10, color=SAMPLE_COLOR, zorder=5)
    axes_exact[2].set_xlabel(r"$y_1$")
    axes_exact[2].set_ylabel(r"$y_2$")
    axes_exact[2].set_title("Exact Output Sets")
    add_reachability_legend(
        axes_exact[2],
        exact_labels=output_exact_labels,
        sample_label="Output samples",
    )
    apply_plot_bounds(axes_exact[2], *output_plot_bounds)

    exact_figure_path = figure_dir / "reachability_star_2d_exact_output.png"
    fig_exact.savefig(exact_figure_path, dpi=200)

    # Figure 2: input, over-approximate hidden ReLU Star, and over-approximate output Star.
    fig_over, axes_over = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    fig_over.suptitle("Over-Approximate Star Reachability for a 2D ReLU Network", fontsize=16, fontweight="bold")

    setup_reachability_axis(axes_over[0])
    plot_starv_set(axes_over[0], input_star, color=INPUT_COLOR)
    axes_over[0].scatter(input_samples[0, :], input_samples[1, :], s=10, color="#023047", zorder=5)
    axes_over[0].set_xlabel(r"$x_1$")
    axes_over[0].set_ylabel(r"$x_2$")
    axes_over[0].set_title("Input Set")
    add_plot_labels(axes_over[0], [("Input Star", INPUT_COLOR), ("Samples", "#023047")])
    apply_plot_bounds(axes_over[0], *input_plot_bounds)

    setup_reachability_axis(axes_over[1])
    plot_starv_set(axes_over[1], hidden_overapprox_set, color=OVERAPPROX_COLOR)
    axes_over[1].scatter(hidden_samples[0, :], hidden_samples[1, :], s=10, color=SAMPLE_COLOR, zorder=5)
    axes_over[1].set_xlabel(r"$h_1$")
    axes_over[1].set_ylabel(r"$h_2$")
    axes_over[1].set_title("Over-Approximate Intermediate Set")
    add_reachability_legend(
        axes_over[1],
        overapprox_label="Over-approximate ReLU Star",
        exact_labels=[],
        sample_label="ReLU samples",
    )
    apply_plot_bounds(axes_over[1], *hidden_plot_bounds)

    setup_reachability_axis(axes_over[2])
    plot_starv_set(axes_over[2], output_overapprox_sets, color=OVERAPPROX_COLOR)
    axes_over[2].scatter(output_samples[0, :], output_samples[1, :], s=10, color=SAMPLE_COLOR, zorder=5)
    axes_over[2].set_xlabel(r"$y_1$")
    axes_over[2].set_ylabel(r"$y_2$")
    axes_over[2].set_title("Over-Approximate Output Set")
    add_reachability_legend(
        axes_over[2],
        overapprox_label="Over-approximate output Star",
        exact_labels=[],
        sample_label="Output samples",
    )
    apply_plot_bounds(axes_over[2], *output_plot_bounds)

    overapprox_figure_path = figure_dir / "reachability_star_2d_overapprox_output.png"
    fig_over.savefig(overapprox_figure_path, dpi=200)

    # Figure 3: over-approximate sets first, exact Stars on top.
    fig_combined, axes_combined = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    fig_combined.suptitle("Exact vs Over-Approximate Reachability for a 2D ReLU Network", fontsize=16, fontweight="bold")

    setup_reachability_axis(axes_combined[0])
    plot_starv_set(axes_combined[0], input_star, color=INPUT_COLOR)
    axes_combined[0].scatter(input_samples[0, :], input_samples[1, :], s=10, color="#023047", zorder=5)
    axes_combined[0].set_xlabel(r"$x_1$")
    axes_combined[0].set_ylabel(r"$x_2$")
    axes_combined[0].set_title("Input Set")
    add_plot_labels(axes_combined[0], [("Input Star", INPUT_COLOR), ("Samples", "#023047")])
    apply_plot_bounds(axes_combined[0], *input_plot_bounds)

    setup_reachability_axis(axes_combined[1])
    plot_starv_set(axes_combined[1], hidden_overapprox_set, color=OVERAPPROX_COLOR, zorder=1)
    hidden_combined_labels = plot_numbered_exact_stars(
        axes_combined[1], hidden_exact_sets, label_prefix="Exact ReLU Star", zorder=3
    )
    axes_combined[1].scatter(hidden_samples[0, :], hidden_samples[1, :], s=10, color=SAMPLE_COLOR, zorder=5)
    axes_combined[1].set_xlabel(r"$h_1$")
    axes_combined[1].set_ylabel(r"$h_2$")
    axes_combined[1].set_title("Intermediate Sets")
    add_reachability_legend(
        axes_combined[1],
        overapprox_label="Over-approximate ReLU Star",
        exact_labels=hidden_combined_labels,
        sample_label="ReLU samples",
    )
    apply_plot_bounds(axes_combined[1], *hidden_plot_bounds)

    setup_reachability_axis(axes_combined[2])
    plot_starv_set(axes_combined[2], output_overapprox_sets, color=OVERAPPROX_COLOR, zorder=1)
    output_combined_labels = plot_numbered_exact_stars(
        axes_combined[2], output_exact_sets, label_prefix="Exact output Star", zorder=3
    )
    axes_combined[2].scatter(output_samples[0, :], output_samples[1, :], s=10, color=SAMPLE_COLOR, zorder=5)
    axes_combined[2].set_xlabel(r"$y_1$")
    axes_combined[2].set_ylabel(r"$y_2$")
    axes_combined[2].set_title("Output Sets")
    add_reachability_legend(
        axes_combined[2],
        overapprox_label="Over-approximate output Star",
        exact_labels=output_combined_labels,
        sample_label="Output samples",
    )
    apply_plot_bounds(axes_combined[2], *output_plot_bounds)

    combined_figure_path = figure_dir / "reachability_star_2d_exact_vs_overapprox.png"
    fig_combined.savefig(combined_figure_path, dpi=200)

    plt.close(fig_exact)
    plt.close(fig_over)
    plt.close(fig_combined)

    # Print a short text summary for users running the tutorial from a terminal.
    print("2D Star reachability demo")
    print(f"Input bounds: lb={input_lb}, ub={input_ub}")
    print(f"Input samples: {input_samples.shape[1]}")
    print(f"Exact ReLU Stars: {len(hidden_exact_sets)}")
    print("Over-approximate ReLU Stars: 1")
    print(f"Exact output Stars: {len(output_exact_sets)}")
    print(f"Exact output Star bounds: lb={exact_output_lb}, ub={exact_output_ub}")
    print(f"Over-approximate output Stars: {len(output_overapprox_sets)}")
    print(f"Over-approximate output Star bounds: lb={overapprox_lb}, ub={overapprox_ub}")
    print(f"Saved exact-output figure: {exact_figure_path}")
    print(f"Saved over-approximate-output figure: {overapprox_figure_path}")
    print(f"Saved exact-vs-overapprox figure: {combined_figure_path}")


if __name__ == "__main__":
    main()
