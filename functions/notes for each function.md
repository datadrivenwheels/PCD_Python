### 🔍 `find_min_rss_like_tau`

Identifies the optimal change point (`tau`) in the variance of residuals from a fitted spline model by minimizing a likelihood-based residual sum of squares (RSS) metric. Incorporates a statistical significance test using the Type I error rate `alpha`.

---

### 🧩 `changing_point_p_spline`

Fits a penalized B-spline to the input data and applies `find_min_rss_like_tau` to detect the most probable variance change point. Returns the estimated change point, spline prediction, and likelihood value.

---

### ♻️ `recursive_change_point_detection`

Recursively applies spline fitting and change point detection on input segments to identify multiple variance change points in the data. Useful for capturing complex patterns with several structural shifts.

---

### 📊 `plot_multiple_change_points`

Visualizes the confidence×IoU data along with the penalized spline fit and highlights all detected change points. Useful for diagnostic and presentation purposes.

---

### 📐 `work_distance`

Calculates the **Perception Characteristics Distance (PCD)** by estimating the farthest distance at which the model maintains a minimum confidence×IoU threshold with high reliability, based on change-point-informed variance estimation. Optionally visualizes the result.

---

### 📦 `compute_iou`

Computes the Intersection over Union (IoU) between a ground-truth bounding box and a predicted bounding box, a fundamental metric in object detection accuracy evaluation.
