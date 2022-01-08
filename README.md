# SingleShotDetector

Reimplementation of [SSD architecture][1] for object detection in PyTorch.

**Authors**: Marcin Pastwa, Błażej Huminiecki

_Work in progress_
- [x] Create base architecture for feature extraction
- [x] Create classifiers and merge detections from multiple scales to one tensor
- [ ] Implement loss function
- [ ] Create training loop
- [ ] Train model on datasets used in paper and compare results
---

The main goal of this project is to reimplement and train SSD300 model and compare results
with results in paper. Implementation is based on PyTorch framework and aims to be clear
and easily understandable, therefore we focus on main concepts presented in paper.

![SSD architecture](figures/ssd300_architecture.png)

Figure 1. The architecture of SSD300. Image taken from the [paper][1].

[1]: https://arxiv.org/abs/1512.02325
