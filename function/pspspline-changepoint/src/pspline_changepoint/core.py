from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
from pygam import LinearGAM, s
from scipy.stats import norm

ArrayLike = np.ndarray

def _ensure_column(x: ArrayLike) -> ArrayLike:
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x

def find_min_rss_like_tau(
    X: ArrayLike,
    y: ArrayLike,
    y_pred: ArrayLike,
    alpha: float = 0.05,
) -> Tuple[Optional[float], Optional[float]]:
    """
    依据分段残差对数似然求最小的分割点 tau，并进行极值型阈值检验。

    Parameters
    ----------
    X : (n, 1) array
        自变量；如是一维，将在外部保证列向量形状。
    y : (n,) array
        因变量观测。
    y_pred : (n,) array
        模型拟合值（来自 GAM）。
    alpha : float
        显著性水平。

    Returns
    -------
    tau : float or None
        检验通过的变点位置（以 X 的数值坐标返回），否则 None。
    min_like : float or None
        通过检验时对应的目标函数值，否则 None。
    """
    X = _ensure_column(np.asarray(X))
    y = np.asarray(y).ravel()
    y_pred = np.asarray(y_pred).ravel()

    n = X.shape[0]
    if n != y.shape[0] or n != y_pred.shape[0]:
        raise ValueError("X, y, y_pred 维度不一致")

    if n < 3:
        return None, None

    min_like = np.inf
    tau_val: Optional[float] = None

    for i in range(1, n - 1):  # 避开首尾作为候选
        left = slice(0, i)
        right = slice(i, n)
        # 防止除以零
        RSS_left = i * np.log(np.sum((y[left] - y_pred[left]) ** 2) / max(i, 1))
        RSS_right = (n - i) * np.log(np.sum((y[right] - y_pred[right]) ** 2) / max(n - i, 1))
        likelihood = RSS_left + RSS_right
        if likelihood < min_like:
            min_like = likelihood
            tau_val = float(X[i, 0])

    # 单段基线
    likelihood_n = n * np.log(np.sum((y - y_pred) ** 2) / n)
    delta_n = np.sqrt(max(likelihood_n - min_like, 0.0))

    # 渐近阈值项
    a_n = np.sqrt(2 * np.log(np.log(n))) / np.log(n)
    b_n = (2 * np.log(np.log(n)) + 0.5 * np.log(np.log(np.log(n))) - np.log(np.pi)) / np.log(n)

    stat = a_n * delta_n * np.sqrt(np.log(n)) - b_n * np.log(n)
    crit = -np.log(-np.log((1 - alpha) / 2))

    if tau_val is not None and stat > crit:
        return tau_val, float(min_like)
    else:
        return None, None


def changing_point_p_spline(
    x: ArrayLike,
    y: ArrayLike,
    alpha: float = 0.05,
    n_splines: int = 10,
    spline_order: int = 3,
) -> Tuple[Optional[float], ArrayLike, Optional[float], LinearGAM]:
    """
    用 pyGAM 的惩罚样条拟合 y=f(x)，并在拟合残差上搜索单一变点。

    Returns
    -------
    tau, y_pred, value, gam
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    X = _ensure_column(x)

    gam = LinearGAM(s(0, n_splines=n_splines, spline_order=spline_order, penalties="auto")).fit(X, y)
    y_pred = gam.predict(X)
    tau, value = find_min_rss_like_tau(X, y, y_pred, alpha)
    return tau, y_pred, value, gam


def _index_of_value(x: ArrayLike, val: float) -> int:
    """避免浮点相等比较，用最接近值的索引。"""
    x = np.asarray(x).ravel()
    return int(np.argmin(np.abs(x - val)))


def recursive_change_point_detection(
    x: ArrayLike,
    y: ArrayLike,
    alpha: float = 0.05,
    min_segment_size: int = 130,
    n_splines: int = 10,
    spline_order: int = 3,
) -> List[float]:
    """
    递归二分查找多个变点；每段长度不足 min_segment_size 时停止。
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    n = x.shape[0]

    if n < min_segment_size:
        return []

    tau, y_pred, value, _ = changing_point_p_spline(
        x, y, alpha=alpha, n_splines=n_splines, spline_order=spline_order
    )
    if tau is None:
        return []

    idx = _index_of_value(x, tau)
    # 防止无限递归
    if idx <= 0 or idx >= n - 1:
        return [float(tau)]

    left_points = recursive_change_point_detection(
        x[:idx], y[:idx], alpha=alpha, min_segment_size=min_segment_size,
        n_splines=n_splines, spline_order=spline_order
    )
    right_points = recursive_change_point_detection(
        x[idx:], y[idx:], alpha=alpha, min_segment_size=min_segment_size,
        n_splines=n_splines, spline_order=spline_order
    )
    all_points = sorted(set(left_points + [float(tau)] + right_points))
    return all_points


def compute_iou(ground_truth, predicted) -> float:
    """
    计算两个矩形框（Series 或含四元组）的 IoU。
    期望格式：[x1, y1, x2, y2]，闭区间处理 +1。
    """
    # 支持 pandas.Series 或序列
    gt = np.array([ground_truth.iloc[i] if hasattr(ground_truth, "iloc") else ground_truth[i] for i in range(4)], dtype=float)
    pr = np.array([predicted.iloc[i] if hasattr(predicted, "iloc") else predicted[i] for i in range(4)], dtype=float)

    x1 = max(gt[0], pr[0])
    y1 = max(gt[1], pr[1])
    x2 = min(gt[2], pr[2])
    y2 = min(gt[3], pr[3])

    inter_w = max(0.0, x2 - x1 + 1.0)
    inter_h = max(0.0, y2 - y1 + 1.0)
    inter = inter_w * inter_h

    area_gt = (gt[2] - gt[0] + 1.0) * (gt[3] - gt[1] + 1.0)
    area_pr = (pr[2] - pr[0] + 1.0) * (pr[3] - pr[1] + 1.0)
    denom = max(area_gt + area_pr - inter, 1e-12)
    return float(inter / denom)
