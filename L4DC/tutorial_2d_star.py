from pathlib import Path
import sys
from itertools import product

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt

from StarV.set.star import Star
from StarV.util.plot import plot_star, getVertices
from StarV.util.lp_solver import sample


def plot_2d_star_with_generators(
    c=None,
    generators=None,
    save_path=None,
):
    if c is None:
        c = np.array([0.0, 0.0])

    if generators is None:
        generators = np.array([
            [1.7, 0.0],    # v1
            [-0.35, 1.05], # v2
        ]).T

    c = np.asarray(c, dtype=float)
    G = np.asarray(generators, dtype=float)

    assert c.shape == (2,)
    assert G.shape[0] == 2

    num_generators = G.shape[1]
    pred_lb = np.full(num_generators, -1.2)
    pred_ub = np.full(num_generators, 1.5)
    
    # Build the actual Star set x = c + G * alpha with bounded predicates.
    V = np.column_stack((c, G))
    C = np.vstack((np.eye(num_generators), -np.eye(num_generators)))
    d = np.concatenate((pred_ub, -pred_lb))
    star = Star(V, C, d, pred_lb, pred_ub)

    # Get vertices using StarV utility function (handles general cases).
    vertices = np.asarray(getVertices(star), dtype=float)
    print(f"Vertices of the Star set:\n{vertices}\n")

    # Random interior samples
    samples = sample(star, N=400, lp_solver='linprog', seed=4)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    # Plot the actual Star set using StarV.
    plt.sca(ax)
    start_patch_count = len(ax.patches)
    start_collection_count = len(ax.collections)
    plot_star(star, show=False, color="#13b8a6")
    for patch in ax.patches[start_patch_count:]:
        patch.set_facecolor("#d7f4ef")
        patch.set_edgecolor("#13b8a6")
        patch.set_linewidth(2.5)
        patch.set_alpha(0.95)
        patch.set_zorder(1)
    for collection in ax.collections[start_collection_count:]:
        collection.set_facecolor("#d7f4ef")
        collection.set_edgecolor("#13b8a6")
        collection.set_linewidth(2.5)
        collection.set_alpha(0.95)
        collection.set_zorder(1)

    # Plot samples
    ax.scatter(
        samples[0],
        samples[1],
        s=9,
        color="#4b5563",
        alpha=0.28,
        label="Samples",
        zorder=2,
    )

    # Plot center
    ax.scatter(c[0], c[1], s=120, color="#dc2626", zorder=5)
    ax.text(c[0] + 0.07, c[1] + 0.03, r"$c$", color="#dc2626", fontsize=15)

    # Plot generator arrows
    colors = ["#6d28d9", "#f59e0b", "#2563eb", "#16a34a"]
    for i in range(num_generators):
        v = G[:, i]
        color = colors[i % len(colors)]

        ax.arrow(
            c[0],
            c[1],
            v[0],
            v[1],
            width=0.018,
            head_width=0.12,
            head_length=0.15,
            length_includes_head=True,
            color=color,
            zorder=4,
        )

        label_pos = c + 1.08 * v
        ax.text(
            label_pos[0],
            label_pos[1],
            rf"$v_{i + 1}$",
            color=color,
            fontsize=14,
            fontweight="bold",
        )

    # Plot vertices
    ax.scatter(
        vertices[:, 0],
        vertices[:, 1],
        s=35,
        color="#0f766e",
        zorder=4,
        label="Vertices",
    )

    ax.set_title(r"2D Star Set: $\mathcal{S}=\{c + V\alpha \mid -1.2 \leq \alpha_i \leq 1.5\}$")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    star_handle = plt.Line2D([], [], color="#13b8a6", linewidth=2.5, label="Star set (plot_star)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([star_handle, *handles], ["Star set (plot_star)", *labels], loc="best")

    pad = 0.5
    all_points = np.column_stack([vertices.T, samples, c[:, None]])
    ax.set_xlim(all_points[0].min() - pad, all_points[0].max() + pad)
    ax.set_ylim(all_points[1].min() - pad, all_points[1].max() + pad)

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"Saved figure: {save_path}")
    plt.show()


if __name__ == "__main__":
    # Save the tutorial figure next to this script, matching
    # tutorial_2d_star_reachability.py.
    figure_dir = Path(__file__).resolve().parent / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    center = np.array([0.0, 0.0])

    # Columns are generator directions v1 and v2.
    generators = np.array([
        [1.7, -0.35],
        [0.0,  1.05],
    ])

    plot_2d_star_with_generators(
        c=center,
        generators=generators,
        save_path=figure_dir / "star_generators_2d.png",
    )