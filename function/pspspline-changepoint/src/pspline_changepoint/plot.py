from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
from .core import recursive_change_point_detection, changing_point_p_spline, _ensure_column
from pygam import LinearGAM, s

# 仅在需要绘图时才导入 matplotlib，避免硬依赖
def _plt():
    import matplotlib.pyplot as plt
    return plt

def plot_multiple_change_points(
    x,
    y,
    title: str,
    alpha: float = 0.05,
    show_spline: bool = True,
    n_splines: int = 10,
    spline_order: int = 3,
):
    """
    可视化：散点 +（可选）样条拟合 + 全部变点的竖线。
    """
    plt = _plt()
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    change_points = recursive_change_point_detection(
        x, y, alpha=alpha, n_splines=n_splines, spline_order=spline_order
    )

    X = _ensure_column(x)
    gam = LinearGAM(s(0, n_splines=n_splines, spline_order=spline_order, penalties="auto")).fit(X, y)
    y_pred = gam.predict(X)

    plt.figure(figsize=(12, 6))
    plt.scatter(x, y, label="Detection confidence × IoU", alpha=0.5, s=50)

    if show_spline:
        plt.plot(x, y_pred, label="Penalized Spline Fit", linestyle="--", linewidth=2)

    colors = plt.cm.rainbow(np.linspace(0, 1, max(len(change_points), 1)))
    for i, (tau, color) in enumerate(zip(change_points, colors)):
        plt.axvline(x=tau, color=color, linestyle="-.", linewidth=2, label=f"Var. change point {i+1}")

    plt.xlabel("Distance (m)")
    plt.ylabel("Detection confidence × IoU")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.show()
    return change_points


def work_distance(
    x,
    y,
    change_points: List[float],
    title: str,
    weather: str,
    prob_threshold: float = 0.5,
    y_thresh: float = 0.5,
    show_plot: bool = True,
    n_splines: int = 10,
    spline_order: int = 3,
):
    """
    根据变点分段，估计各段残差波动并计算工作距离（第一个使 P<=阈值 的 x）。
    """
    import numpy as np
    from scipy.stats import norm

    plt = _plt() if show_plot else None

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    X = x.reshape(-1, 1)

    gam = LinearGAM(s(0, n_splines=n_splines, spline_order=spline_order, penalties="auto")).fit(X, y)
    y_spline = gam.predict(X)

    probs = np.ones_like(x, dtype=float)

    if not change_points:
        sigma = float(np.std(y))
        if sigma > 0:
            probs = 1 - norm.cdf(y_thresh, loc=y_spline, scale=sigma)
    else:
        cps = sorted(change_points)
        segments = []
        segments.append(x <= cps[0])
        for i in range(len(cps) - 1):
            segments.append((x >= cps[i]) & (x <= cps[i + 1]))
        segments.append(x >= cps[-1])

        for mask in segments:
            if np.any(mask):
                sigma = float(np.std(y[mask]))
                scale = sigma if sigma > 0 else 1e-10
                probs[mask] = 1 - norm.cdf(y_thresh, loc=y_spline[mask], scale=scale)

    valid_x = x[probs <= prob_threshold]
    max_x = float(np.min(valid_x)) if valid_x.size > 0 else None

    if show_plot and plt is not None:
        plt.figure(figsize=(12, 6))
        plt.scatter(x, y, label="IoU × Confidence score", alpha=0.5, s=50)
        plt.plot(x, y_spline, label="Penalized Spline Fit", linestyle="--", linewidth=2)
        plt.axhline(y=prob_threshold, linestyle="-.", linewidth=2, label=f"y_t = {prob_threshold}")
        if max_x is not None:
            plt.axvline(x=max_x, color="red", linestyle=":", linewidth=2, label="PCD")
        plt.title(f"{title}: {weather}")
        plt.xlabel("Distance (m)")
        plt.ylabel("IoU × Confidence score")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.show()

    return max_x
