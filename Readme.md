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
| `PSPNet.py` | PSPNet | Pyramid Pooling Module (PPM) | None |
| `PSPNet-ChannelAttention.py` | PSPNet-CA | PSPNet | Channel Attention |
| `PSPNet-SpatialAttention.py` | PSPNet-SA | PSPNet | Spatial Attention |
