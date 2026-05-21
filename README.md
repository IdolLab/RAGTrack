<p align="center">
  <img src="https://img.shields.io/badge/CVPR-2026-5B4CF6?style=for-the-badge&logo=openaccess&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.3+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
</p>

<h1 align="center">
  <a href="https://arxiv.org/abs/2603.03617">RAGTrack: Language-aware RGBT Tracking<br>with Retrieval-Augmented Generation</a>
</h1>

<p align="center">
  <a href="https://cvpr.thecvf.com/virtual/2026/poster/37117"><strong>CVPR 2026</strong></a>
</p>

<p align="center">
  <a href="https://orcid.org/0009-0009-2668-7908">Hao Li</a>, 
  <a href="https://924973292.github.io/">Yuhao Wang</a>, 
  <a href="https://orcid.org/0000-0002-1526-7889">Wenning Hao</a>📧, 
  <a href="https://scholar.google.com/citations?user=MfbIbuEAAAAJ&hl=zh-CN">Pingping Zhang</a>📧, 
  <a href="https://scholar.google.com/citations?user=nVgPQpoAAAAJ&hl=zh-CN">Dong Wang</a>, 
  <a href="https://scholar.google.com/citations?user=D3nE0agAAAAJ&hl=zh-CN">Huchuan Lu</a>
</p>

<p align="center">
  <b>
    <a href="https://arxiv.org/abs/2603.03617">📄 Paper</a> &nbsp;|&nbsp;
    <a href="https://github.com/IdolLab/RAGTrack">💻 Code</a> &nbsp;|&nbsp;
    <a href="https://pan.baidu.com/s/1MiRG2wMaHMdNPo4-U52ENw?pwd=3ure">🤖 Models</a> &nbsp;|&nbsp;
    <a href="https://pan.baidu.com/s/1wE2XaOgTkcTIED6Xcma5VA?pwd=maa5">📊 Results</a>
  </b>
</p>

---

## 📝 Abstract

<p align="justify">
  This repository contains the official implementation of <strong>RAGTrack</strong>, the first language-aware RGBT tracking framework powered by Retrieval-Augmented Generation (RAG). We introduce textual descriptions into RGBT benchmarks via MLLM-based annotation pipelines, and propose a novel framework consisting of a Multi-modal Transformer Encoder (MTE), Adaptive Token Fusion (ATF), and Context-aware Reasoning Module (CRM). Included are training/evaluation <a href="https://github.com/IdolLab/RAGTrack">codes</a>, <a href="https://pan.baidu.com/s/1MiRG2wMaHMdNPo4-U52ENw?pwd=3ure">models</a>, and <a href="https://pan.baidu.com/s/1wE2XaOgTkcTIED6Xcma5VA?pwd=maa5">results</a>.
</p>

---

## 🔥 Motivation

<p align="center">
  <img src="assets/motivation.jpg" width="85%" alt="RAGTrack Motivation">
  <br>
  <em>Figure 1. (a) Existing RGBT trackers suffer from inadequate appearance modeling, search redundancy, and modality gap. 
  (b) Our RAGTrack introduces linguistic reasoning, dynamic token selection, and adaptive channel exchange for robust tracking.</em>
</p>

---

## 🏗️ Framework

<p align="center">
  <img src="assets/pipline.jpg" width="95%" alt="RAGTrack Pipeline">
  <br>
  <em>Figure 2. Overall framework of RAGTrack. MTE performs unified visual-language modeling, ATF dynamically selects target-relevant tokens and enables adaptive channel exchange, and CRM retrieves relevant contexts for context-aware reasoning.</em>
</p>

---

## 🔬 Adaptive Token Fusion (ATF)

<p align="center">
  <img src="assets/ATCF.jpg" width="90%" alt="Adaptive Token Fusion">
  <br>
  <em>Figure 3. Details of ATF. Dynamic token selection leverages text-guided attention scores to retain target-relevant tokens, while adaptive channel exchange bridges heterogeneous modality gaps.</em>
</p>

---

## ⚙️ Installation
Create and activate a conda environment:
```
cd path/to/RAGTrack
conda create -n RAGTrack python=3.10
conda activate RAGTrack
```
Download [mamba_install](https://pan.baidu.com/s/1Uy1cICsuEKUgv5eMODn5Aw?pwd=a4ja) and install the required packages:
```
bash install_cadtrack.sh
```

## 📂 Data Preparation
Download the following datasets and place them under ./data/:
- [GTOT, RGBT210, RGBT234, LasHeR](https://chenglongli.cn/Datasets-and-benchmark-code/)
- [VTUAV](https://zhang-pengyu.github.io/DUT-VTUAV/)
```
$<PATH_of_CADTrack>
-- data
    -- GTOT
        |-- BlackCar
        |-- Black5wan1
        ...
    -- RGBT210
        |-- afterrain
        |-- aftertree
        ...
    -- RGBT234
        |-- afterrain
        |-- aftertree
        ...
    -- LasHeR/train
        |-- 1boygo
        |-- 1handsth
        ...
    -- LasHeR/test
        |-- 1blackteacher
        |-- 1boycoming
        ...
    -- VTUAV/train
        |-- animal_002
        |-- bike_002
        ...
    -- VTUAV/test_ST
        |-- animal_001
        |-- bike_003
        ...
    -- VTUAV/test_LT
        |-- animal_003
        |-- animal_004
        ...
```

## 🔧 Setup & Configuration
Run the following command to set paths:
```
cd <PATH_of_CADTrack>
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
You can also modify paths by these two files:
```
./lib/train/admin/local.py  # paths for training
./lib/test/evaluation/local.py  # paths for testing
```

## 🏋️ Training
Download the [pretrained model](https://pan.baidu.com/s/15GjTLQboXcfJaTD5sLLRDQ?pwd=hmaa) and put it under ./pretrained/.
```
bash train.sh
```
You can train models with various variants by modifying ```train.sh```.

## 📊 Testing
### Testing on Benchmark Datasets
Modify the <DATASET_PATH> and <SAVE_PATH> in```./RGBT_workspace/test_rgbt_mgpus.py```, then run:
```
bash test.sh
```

### Evaluation Tools
- GTOT/RGBT210/RGBT234/LasHeR: Use the [Evaluation Toolkit](https://chenglongli.cn/Datasets-and-benchmark-code/)
- VTUAV: Follow the [VTUAV_Evaluation](https://zhang-pengyu.github.io/DUT-VTUAV/)

## 🖼️ Poster

<p align="center">
  <img src="assets/poster.jpg" width="85%" alt="CVPR 2026 Poster">
</p>

## 📝 Citation
If you find CADTrack is helpful for your research, please consider citing:

```bibtex
@inproceedings{li2026ragtrack,
  title={RAGTrack: Language-aware RGBT Tracking with Retrieval-Augmented Generation},
  author={Li, Hao and Wang, Yuhao and Hao, Wenning and Zhang, Pingping and Wang, Dong and Lu, Huchuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

## 🙏 Acknowledgments
 This repo is based on [STTrack](https://github.com/NJU-PCALab/STTrack) and [IDEA](https://github.com/924973292/IDEA) which are excellent works.
<p align="center"> <b>Star ⭐ this repo if you like our work!</b> </p>
