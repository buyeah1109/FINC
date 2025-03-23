# FINC: Fourier-based Differential Clustering

This repository provides an official implementation of CVPR'25 paper:

 "Unveiling Differences in Generative Models: A Scalable Differential Clustering Approach"

Authors: Jingwei Zhang, Mohammad Jalali, Cheuk Ting Li, Farzan Farnia

## Dependency
<ol>
    <li> Framework: PyTorch, torchvision </li>
    <li> Encoders: clip, torchvision </li>
    <li> Progress bar: tqdm </li>
</ol>

## Quick Access
[Code Descriptions](#quick-start) <br>

## Quick Start
Pipeline:
<ol>
    <li> Initialize FINC evaluator </li>
    <li> Select feature extractor (Currently support <a href='https://arxiv.org/abs/1512.00567'> Inception-V3</a>, <a href='https://arxiv.org/abs/2304.07193'> DINOv2</a>, <a href='https://arxiv.org/abs/2103.00020'> CLIP</a>, <a href='https://arxiv.org/abs/2006.09882'> SwAV</a>, ResNet50)
    <li> Detect novel modes by FINC</li>
</ol>

<br>

```python 
from src.metric.FINC import FINC_Evaluator

# Core object for detect novel modes by FINC
evaluator = FINC_Evaluator(logger_path: str, # Path to save log file
                          batchsize: int, # Batch size
                          sigma: int, # Bandwidth parameter in RBF kernel
                          eta: int, # Novelty threshold
                          num_samples: int, # Sampling number for EACH distribution
                          result_name: str, # Unique name for saving results
                          rff_dim: int) # random fourier features dimension to approximate kernel

# Select feature extractor
evaluator.set_feature_extractor(name: str = 'dinov2', # feature extractor ['inception', 'dinov2', 'clip', 'resnet50', 'swav]
                                save_path: str | None = './save') # Path to save calculated features for reuse

# Extract novel modes of novel_dataset w.r.t. reference dataset by FINC
FINC.rff_differential_clustering_modes_of_dataset(novel_dataset: torch.utils.Dataset,
                                                  ref_dataset: torch.utils.Dataset)
```

## Examples for More Functionality
### Save Extracted Features and Indexes for Further Use
In some cases, we may save the extracted features to reduce repeating computation (e.g. tuning bandwidth parameter, novelty threshold). We may specify the folder to save and load features:
```python
evaluator.set_feature_extractor(name = 'dinov2', # feature extractor ['inception', 'dinov2', 'clip', 'swav', 'resnet50']
                                save_path = './save') # Path to save calculated features for reuse
```
In this example, the evaluator will first check whether './save/dinov2/[result_name]_[other_information].pt' exists. If not, the evaluator will extract features and their indexes in the dataset, and save to this path.
