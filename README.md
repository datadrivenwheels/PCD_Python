# Perception Characteristics Distance (PCD)

**“Perception Characteristics Distance: Measuring Stability and Robustness of Perception System in Dynamic Conditions under a Certain Decision Rule”**
Boyu Jiang, Liang Shi, Zhengzhi Lin, Loren Stowe, Feng Guo (Jun 10, 2025) ([arxiv.org][1])

This repository provides the official Python implementation of the Perception Characteristics Distance (PCD), a metric designed to evaluate the reliable detection range of perception systems under dynamic real-world conditions (e.g. varying weather), along with the associated **SensorRainFall** dataset.

---

## 🚗 Abstract

The performance of perception systems in autonomous driving systems (ADS) is strongly influenced by object distance, scene dynamics, and environmental conditions such as weather. AI-based perception outputs are inherently stochastic, with variability driven by these external factors, while traditional evaluation metrics remain static and event-independent, failing to capture fluctuations in confidence over time. In this work, we introduce the Perception Characteristics Distance (PCD) -- a novel evaluation metric that quantifies the farthest distance at which an object can be reliably detected, incorporating uncertainty in model outputs. To support this, we present the SensorRainFall dataset, collected on the Virginia Smart Road using a sensor-equipped vehicle (cameras, radar, LiDAR) under controlled daylight-clear and daylight-rain scenarios, with precise ground-truth distances to the target objects. Statistical analysis reveals the presence of change points in the variance of detection confidence score with distance. By averaging the PCD values across a range of detection quality thresholds and probabilistic thresholds, we compute the mean PCD (mPCD), which captures the overall perception characteristics of a system with respect to detection distance. Applying state-of-the-art perception models shows that mPCD captures meaningful reliability differences under varying weather conditions -- differences that static metrics overlook. PCD provides a principled, distribution-aware measure of perception performance, supporting safer and more robust ADS operation, while the SensorRainFall dataset offers a valuable benchmark for evaluation. 

---


## 🎯 Key Features

* 🔍 **PCD Computation**: Implements heteroscedastic modeling of IoU×confidence vs. distance using penalized B‑spline regression and variance change-point detection ([themoonlight.io][2]).
* 📈 **mPCD Evaluation**: Iterates across (y\_t, p\_t) threshold pairs to compute mPCD, enabling a comprehensive reliability profile.
* ☔ **SensorRainFall Support**: Processes data from clear and rainy driving scenarios for performance analysis.
* 📊 **Model Comparison**: Facilitates evaluation of various object detection models (e.g., Deformable DETR, YOLOX) under different environmental conditions ([arxiv.org][3]).

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

* SensorRainFall dataset is available here: [https://www.kaggle.com/datasets/datadrivenwheels/sensorrainfall](https://www.kaggle.com/datasets/datadrivenwheels/sensorrainfall) ([arxiv.org][1])


---

## 🧪 Reproducing Paper Results

The benchmarks in the paper compare five leading models (Deformable DETR, Grounding DINO, DyHead, YOLOX, GLIP) under both clear and rainy conditions, showcasing distinct reliability degradations captured by mPCD that remain hidden to static metrics ([github.com][4], [arxiv.org][3]).

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

## 📌 License

Shared under the [MIT License](LICENSE). Refer to `requirements.txt` for third-party license details.

---

## ⚡ Contact

For questions, issues, or contributions, open an issue or contact the maintainers via GitHub.

---

**Enjoy evaluating perception robustness with PCD!**

[1]: https://arxiv.org/abs/2506.09217?utm_source=chatgpt.com "Perception Characteristics Distance: Measuring Stability and Robustness of Perception System in Dynamic Conditions under a Certain Decision Rule"
[4]: https://github.com/Kaggle/kaggle-api/blob/master/kaggle/api/kaggle_api_extended.py?utm_source=chatgpt.com "kaggle-api/kaggle/api/kaggle_api_extended.py at main - GitHub"
