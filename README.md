
# FGBENet: Frequency-domain Global Boundary Enhancement Network

## 🌱 Introduction
**FGBENet** (Frequency-domain Global Boundary Enhancement Network) is a multitask deep learning framework for **agricultural parcel delineation** from high-resolution remote sensing imagery.  
It integrates two novel modules:

1. **Frequency-domain Global Gated Attention (FGGA)**  
   Enhances long-range dependency modeling and cross-scale consistency by leveraging frequency-domain attention with gating.  

2. **Gated Boundary Enhancement (GBE)**  
   Refines fine-grained boundaries with gated convolutions, improving the clarity and continuity of ridges, ditches, and crop rows.  

<p align="center">
  <img width="800" alt="FGBENet-architecture" src="https://github.com/user-attachments/assets/5c032d56-993d-42a4-8e2a-e3dc1d7c30e0" />
</p>

---

## ⚙️ Installation
Clone this repository:
```bash
git clone https://github.com/liye1213/FGBENET.git
cd FGBENET
## 🚀 Usage
Run the following example to test the model:

```python
import torch
from FGBENet import Field

model = Field()
x = torch.randn(1, 3, 256, 256)

outputs = model(x)
mask, edge, distance = outputs
## 📜 Citation
If you find this work useful, please cite our paper:

```bibtex
@inproceedings{li2025fgbenet,
  title={FGBENet: Frequency-domain Global Boundary Enhancement Network for Agricultural Parcel Delineation},
  author={Ye Li and Liruizhi Jia and Chao Liu and Bo Kong and Yuan Liu and Shengquan Liu},
  booktitle={ICASSP},
  year={2025}
}

