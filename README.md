# MOSAI-Net

**MOSAI-Net: MultimOdal Selection and Adaptive Interaction on Graph Networks for Cancer Recurrence Prediction**

## Overview

MOSAI-Net is a graph neural network framework for cancer recurrence prediction using multimodal clinical data.  
It effectively handles modality incompleteness while preserving modality specificity via a Multimodal Complementary Graph Builder, an Adaptive Modality Selection Fusion Module, and a Dynamic Graph Structure Optimization Network.

---

## Prerequisites

This project is implemented using **two separate environments** due to differing dependencies between modules.

---

### Environment 1: Graph Construction (`Multimodal Complementary Graph Builder`)

This environment is used for multimodal data preprocessing and patient relationship graph construction.

**Requirements**

- Python ≥ 3.10.16 
- PyTorch ≥ 2.7.0
- Transformers ≥ 4.52.4
- NumPy ≥ 1.24.3
- SciPy ≥ 1.13.1
- Scikit-learn ≥ 1.6.1


### Environment 2: Model Training (`Adaptive Modality Selection Fusion Module` & `Dynamic Graph Structure Optimization Network`)

This environment is used for model training, adaptive modality selection fusion (AMSF),  
and dynamic graph structure optimization (DGSO).

**Requirements**

- Python ≥ 3.9
- PyTorch == 1.10.2
- NumPy ≥ 1.22.4
- SciPy ≥ 1.8.1
- Scikit-learn ≥ 1.6.1

---

## Data
The clinical dataset used in this study is not publicly available due to patient privacy
and institutional ethical regulations.

Researchers interested in accessing the data may contact the authors.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/bioinformatics-xu/MOSAI-Net.git
cd MOSAI-Net
```
---

## Usage

### Step 1: Graph Construction

Run the graph building script to construct the patient relationship graph:

```bash
cd build_graph
python creat_graph.py
```

### Step 2: Model Training

Train MOSAI-Net using the generated graph data:
```bash
cd model
python train_cv.py
```

### Experimental Settings

All experiment-related configurations, scripts, and evaluation procedures are provided in:
./experiment

---

## Developers

**Xiaolu Xu**  
📧 lu.xu@lnnu.edu.cn  

**Shuai Zheng**  
📧 zhengshuai714@163.com  

School of Computer and Artificial Intelligence  
Liaoning Normal University
