# Detecting Linear Deformation Features in Arctic Landfast Ice Using Radar Interferometry
## Project Overview
This project aims to automatically and accurately extract and identify linear deformation features (such as cracks, shear zones, etc.) in Arctic Landfast Ice using Radar Interferometry (InSAR) technology combined with advanced deep learning semantic segmentation models. This repository contains implementations of nine deep learning models for polar ice condition analysis, along with detailed configurations and experimental performance records.
## Model Architectures
This study systematically compares three mainstream semantic segmentation network architectures (U-Net, DeepLabV3, PSPNet) and their variants incorporating Channel Attention (CA) and Spatial Attention (SA) mechanisms. All models are implemented in PyTorch or compatible frameworks.
| Filename | Model Name | Core Architecture | Attention Mechanism |
|----------|------------|-------------------|---------------------|
| `Unet.py` | U-Net | Encoder-Decoder Structure | None |
| `Unet-ChannalAttention.py` | U-Net-CA | U-Net | Channel Attention |
| `Unet-SpatialAttention.py` | U-Net-SA | U-Net | Spatial Attention |
| `DeepLabV3.py` | DeepLabV3 | Atrous Spatial Pyramid Pooling (ASPP) | None |
| `DeepLabV3ChannelAttention.py` | DeepLabV3-CA | DeepLabV3 | Channel Attention |
| `DeepLabV3SpatialAttention.py` | DeepLabV3-SA | DeepLabV3 | Spatial Attention |
| `PSPNet.py` | PSPNet | ASPP | None |
| `PSPNet-ChannelAttention.py` | PSPNet-CA | PSPNet | Channel Attention |
| `PSPNet-SpatialAttention.py` | PSPNet-SA | PSPNet | Spatial Attention |

Experimental Results
| Model | mIoU (%) | mPA (%) | F1-Score (%) | OA (%) |
|-------|---------:|--------:|-------------:|-------:|
| U-Net | 74.03 | 79.18 | 82.78 | 96.44 |
| U-Net-CA | 76.09 | 80.93 | 83.60 | 96.88 |
| U-Net-SA | 75.15 | 79.26 | 83.72 | 96.73 |
| DeepLabV3 | 67.40 | 72.00 | 75.18 | 95.72 |
| DeepLabV3-CA | 66.90 | 70.92 | 75.87 | 95.52 |
| DeepLabV3-SA | 67.48 | 72.42 | 76.49 | 95.42 |
| PSPNet | 66.51 | 70.42 | 75.46 | 95.49 |
| PSPNet-CA | 66.54 | 70.92 | 75.50 | 95.39 |
| PSPNet-SA | 65.72 | 69.65 | 74.61 | 95.35 |
