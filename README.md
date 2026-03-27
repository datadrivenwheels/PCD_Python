# Perception Characteristics Distance (PCD) - CVPR 2026

**“Perception Characteristics Distance: Measuring Stability and Robustness of Perception System in Dynamic Conditions under a Certain Decision Rule”**
Boyu Jiang, Liang Shi, Zhengzhi Lin, Lanxin Xiang, Loren Stowe, Feng Guo ([ArXiv][1])

This repository provides the official Python implementation of the Perception Characteristics Distance (PCD), a metric designed to evaluate the reliable detection range of perception systems under dynamic real-world conditions (e.g. varying weather), along with the associated [**SensorRainFall**][2] dataset.

---

## 🚗 Abstract

The safety of autonomous driving systems (ADS) depends on accurate perception across distance and driving conditions. The outputs of AI perception algorithms are stochastic, which has a major impact on decision making and safety outcomes, including time-to-collision estimation. However, current perception evaluation metrics do not reflect the stochastic nature of perception algorithms. We therefore introduce the Perception Characteristics Distance (PCD), a novel metric incorporating model output uncertainty as represented by the farthest distance at which an object can be reliably detected. To represent a system’s overall perception capability in terms of reliable detection distance, we used the averaged PCD values across multiple detection quality and probabilistic thresholds to produce the average PCD (aPCD). For empirical validation, we present the SensorRainFall dataset, collected on the Virginia Smart Roads using a sensor-equipped vehicle (cameras, radar, and LiDAR) controlled under different weather (clear and rainy) and illumination conditions (daylight, streetlight, and night). The dataset includes ground-truth distances, bounding boxes, and segmentation masks for target objects. Experiments with state-of-theart models show that aPCD captures meaningful differences across weather, daylight, and illumination conditions, which traditional evaluation metrics fail to reflect. PCD provides an uncertainty-aware measure of perception performance, supporting safer and more robust ADS operation, while the SensorRainFall dataset offers a valuable benchmark for evaluation. 

---


## 🎯 Key Features

* 🔍 **PCD Computation**: Implements heteroscedastic modeling of IoU×confidence vs. distance using penalized B‑spline regression and variance change-point detection.
* 📈 **aPCD Evaluation**: Iterates across threshold pairs ($y^thres$, $p^thres$) to compute aPCD, enabling a comprehensive reliability profile.
* ☔ **SensorRainFall Support**: Processes data from clear and rainy driving scenarios for performance analysis.
* 📊 **Model Comparison**: Facilitates evaluation of various object detection, instance segmentation, and object tracking models (e.g., Deformable DETR, YOLOX) under different environmental conditions.

---

## 📚 Getting Started

### Prerequisites

* Python 3.8+
* Dependencies listed in `requirements.txt`

Install required packages:

```bash
pip install -r requirements.txt
```

### Data Preparation

* [**SensorRainFall**][2] dataset is available in Kaggle.


---

## 🎓 Citation

If using this work, please cite:

```
@misc{jiang2025perceptioncharacteristicsdistancemeasuring,
      title={Perception Characteristics Distance: Measuring Stability and Robustness of Perception System in Dynamic Conditions under a Certain Decision Rule}, 
      author={Boyu Jiang and Liang Shi and Zhengzhi Lin and Loren Stowe and Feng Guo},
      year={2025},
      eprint={2506.09217},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2506.09217}, 
}
```

---

**Enjoy evaluating perception robustness with PCD!**

[1]: https://arxiv.org/abs/2506.09217
[2]: https://www.kaggle.com/datasets/datadrivenwheels/sensorrainfall
