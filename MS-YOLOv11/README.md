# MS-YOLOv11
This repository contains the official implementation of the paper:  
> **Lightweight Road Defect Detection via Multi-layer Feature Fusion and Parameter Sharing**  
> Submitted to *The Visual Computer* (2026).

-------
## 📌 Overview
MS-YOLOv11 is a highly efficient, lightweight visual detection framework designed for robust road surface defect detection (such as cracks and potholes) in complex, unstructured environments. 

To overcome the theoretical bottlenecks of spatial aliasing and semantic attenuation in traditional networks, our framework introduces several key innovations:
- **Multi-layer Feature Fusion Diffusion (MFFD) Pyramid**: Replaces the conventional neck to comprehensively aggregate multi-scale defect representations.
- **Large-Kernel Context Downsampling (LKC-Down)**: Preserves high-frequency fine defect details while rigorously suppressing spatial feature aliasing.
- **Split-Rep-Feature Enhancement Block (SRFE-Block)**: Integrates frequency-domain feature decoupling and SimAM attention to dynamically amplify salient defect textures while filtering complex background clutter.
- **Shared Split-Feature Detection Head (SSDH)**: A "shared-then-decoupled" paradigm that drastically reduces parameter count while maintaining the latent synergy between spatial localization and semantic classification.

The proposed architecture achieves a Pareto optimal balance, yielding high accuracy (**82.9% mAP@0.5** on RDD2022) with extreme lightweight efficiency (**2.10M parameters**) and real-time inference (**300 FPS**).

-------
## ⚙️ Repository Structure
MS-YOLOv11/
│-- data/                # Dataset configuration files (e.g., RDD2022.yaml)
│-- models/              # Core architectural modifications (LKC-Down, SSDH, SRFE-Block)
│-- weights/             # Directory for pre-trained weights (e.g., ms-yolov11-best.pt)
│-- requirements.txt     # List of dependencies for environment setup
│-- train.py             # Training script
│-- val.py               # Validation script
│-- detect.py            # Inference demo script 
│-- README.md            # Project documentation


##  📂 Dataset Availability

A **partial dataset** is included in the `data/` folder to demonstrate the usage of the provided scripts.  

> ⚠️ **Note:**  
> Due to confidentiality requirements and the fact that the project is still ongoing, the **full dataset cannot be released at this stage**.  
> Upon completion of the project, the full dataset will be uploaded to this repository .  

---
## 📖 Citation

 If you find this work useful, please cite our manuscript:
 @article{MS-YOLOv11,
  title   = {Lightweight Road Defect Detection via Multi-layer Feature Fusion and Parameter Sharing},
  author  = {Zhang Miao and Wu Huan and Lei Jingfa},
  journal = {The Visual Computer},
  year    = {2026}
}


 ## 🔗 Related Information
This repository is directly associated with the manuscript submitted to The Visual Computer.
