from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt

def plot_relu_exact_split(l=-2.0, u=3.0, save_path="relu_exact_split.png"):
    """
    Figure 1:
    Exact reachability for ReLU using linear split into two cases:
      1) x in [l, 0]  -> y = 0
      2) x in [0, u]  -> y = x
    """
    assert l < 0 < u, "Need l < 0 < u for split case."

    x = np.linspace(l - 0.5, u + 0.5, 500)
    y = np.maximum(0, x)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot ReLU curve
    ax.plot(x, y, linewidth=2.5, label=r"$y=\mathrm{ReLU}(x)=\max(0,x)$")

    # Exact split branches
    x_neg = np.linspace(l, 0, 100)
    y_neg = np.zeros_like(x_neg)
    x_pos = np.linspace(0, u, 100)
    y_pos = x_pos

    ax.plot(x_neg, y_neg, linewidth=4, label=r"Exact branch 1: $x\in[l,0],\ y=0$")
    ax.plot(x_pos, y_pos, linewidth=4, label=r"Exact branch 2: $x\in[0,u],\ y=x$")

    # Mark interval endpoints and split point
    ax.scatter([l, 0, u], [0, 0, u], s=70, zorder=5)

    # Input interval on x-axis
    ax.hlines(0, l, u, linewidth=5, alpha=0.25, label=r"Input set: $x\in[l,u]$")

    # Annotations
    ax.axvline(0, linestyle="--", linewidth=1.5)
    ax.text(l, -0.35, f"l={l}", ha="center", fontsize=11)
    ax.text(0, -0.35, "0", ha="center", fontsize=11)
    ax.text(u, -0.35, f"u={u}", ha="center", fontsize=11)

    ax.set_title("Exact ReLU Reachability\nSplit into two linear regions", fontsize=16)
    ax.set_xlabel("Input x", fontsize=13)
    ax.set_ylabel("Output y", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc="upper left")
    ax.set_xlim(l - 0.5, u + 0.5)
    ax.set_ylim(-0.5, u + 0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_relu_triangle_overapprox(l=-2.0, u=3.0, save_path="relu_triangle_overapprox.png"):
    """
    Figure 2:
    Triangular over-approximation of ReLU over interval [l, u], l < 0 < u.

    Convex relaxation:
        y >= 0
        y >= x
        y <= (u/(u-l)) * (x - l)
    """
    assert l < 0 < u, "Need l < 0 < u for triangular relaxation."

    x = np.linspace(l - 0.5, u + 0.5, 500)
    y = np.maximum(0, x)

    # Upper relaxation line
    slope = u / (u - l)
    upper_line = slope * (x - l)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Exact ReLU graph
    ax.plot(x, y, linewidth=2.5, label=r"Exact ReLU graph")

    # Triangle vertices: (l,0), (0,0), (u,u)
    tri_x = np.array([l, 0, u])
    tri_y = np.array([0, 0, u])

    # Fill over-approximation triangle
    ax.fill(tri_x, tri_y, alpha=0.25, label="Triangular over-approximation")

    # Plot triangle boundaries
    ax.plot([l, u], [0, u], linewidth=2, label=r"Upper bound: $y \leq \frac{u}{u-l}(x-l)$")
    ax.plot([l, u], [0, 0], linewidth=2, linestyle="--", label=r"Lower bound: $y \geq 0$")
    ax.plot([l, u], [l, u], linewidth=2, linestyle="--", label=r"Lower bound: $y \geq x$")

    # Mark key points
    ax.scatter([l, 0, u], [0, 0, u], s=70, zorder=5)

    ax.axvline(0, linestyle="--", linewidth=1.2)
    ax.text(l, -0.35, f"l={l}", ha="center", fontsize=11)
    ax.text(0, -0.35, "0", ha="center", fontsize=11)
    ax.text(u, -0.35, f"u={u}", ha="center", fontsize=11)

    ax.set_title("Triangular ReLU Over-Approximation\nOne convex set encloses the exact graph", fontsize=16)
    ax.set_xlabel("Input x", fontsize=13)
    ax.set_ylabel("Output y", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_xlim(l - 0.5, u + 0.5)
    ax.set_ylim(l - 0.3, u + 1.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # Save the tutorial figure next to the script.
    figure_dir = Path(__file__).resolve().parent / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    plot_relu_exact_split(l=-2.0, u=3.0, save_path=figure_dir / "relu_exact_split.png")
    plot_relu_triangle_overapprox(l=-2.0, u=3.0, save_path=figure_dir / "relu_triangle_overapprox.png")