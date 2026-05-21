<h1 align="center">🎯 CADTrack: Learning Contextual Aggregation with Deformable Alignment for Robust RGBT Tracking</h1>

![motivation](assets/motivation.jpg)

<div align="center">
  <a href="https://arxiv.org/abs/2511.17967">CADTrack: Learning Contextual Aggregation with Deformable Alignment for Robust RGBT Tracking</a><br>
  <a href="https://orcid.org/0009-0009-2668-7908">Hao Li</a>, 
  <a href="https://924973292.github.io/">Yuhao Wang</a>, 
  <a href="https://xiantaohu.github.io/index.html">Xiantao Hu</a>, 
  <a href="https://orcid.org/0000-0002-1526-7889">Wenning Hao*</a>, 
  <a href="https://scholar.google.com/citations?user=MfbIbuEAAAAJ&hl=zh-CN">Pingping Zhang*</a>, 
  <a href="https://scholar.google.com/citations?user=nVgPQpoAAAAJ&hl=zh-CN">Dong Wang</a>, 
  <a href="https://scholar.google.com/citations?user=D3nE0agAAAAJ&hl=zh-CN">Huchuan Lu</a><br>
  <a href="https://ojs.aaai.org/index.php/AAAI/article/view/37535"><strong>AAAI 2026</strong></a>
</div>

<p align="justify">
This repository contains the official implementation of <a href="https://arxiv.org/pdf/2511.17967"><strong>CADTrack</strong></a>, a novel framework for robust RGB-Thermal (RGBT) object tracking. CADTrack addresses key challenges of modality discrepancies and spatial misalignment via three innovative components: <strong>Mamba-based Feature Interaction (MFI)</strong> for efficient cross-modal interaction, <strong>Contextual Aggregation Module (CAM)</strong> for dynamic multi-layer feature fusion, and <strong>Deformable Alignment Module (DAM)</strong> for spatiotemporal alignment. Included are training/evaluation <a href="https://github.com/IdolLab/CADTrack">codes</a>, <a href="https://pan.baidu.com/s/1GhiRfekS90EFsEePTHbzYg?pwd=3mbq">models</a>, and <a href="https://pan.baidu.com/s/1GhiRfekS90EFsEePTHbzYg?pwd=3mbq">results</a>.
</p>

## 🚀 New
- 🎉 Paper Accepted at AAAI 2026!
- 📦 Code & Models Released – Full implementation now publicly available.
  
## ✨ Key Features & Contributions
<p align="center">
  <p align="center">
    <img src="assets/pipline.jpg" alt="Description of the image" style="width:100%;">
  <p align="center">
<p align="center" style="font-size: 18px; color: gray;">
    Figure 1: Overall framework of CADTrack.
</p>
<p align="center">
    <img src="assets/mambabridge.jpg" alt="RGBNT201 assets" style="width:100%;">
</p>
<p align="center" style="font-size: 18px; color: gray;">
    Figure 2: Details of MFI.
</p>
<p align="center">
    <img src="assets/moe.jpg" alt="RGBNT201 assets" style="width:100%;">
</p>
<p align="center" style="font-size: 18px; color: gray;">
    Figure 3: The structure of CAM.
</p>
<p align="center">
    <img src="assets/offset.jpg" alt="RGBNT201 assets" style="width:100%;">
</p>
<p align="center" style="font-size: 18px; color: gray;">
    Figure 4: Deformable alignment of DAM.
</p>

## ⚙️ Installation
Create and activate a conda environment:
```
cd path/to/CADTrack
conda create -n CADTrack python=3.10
conda activate CADTrack
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

## 📜 Poster
<p align="center">
    <img src="assets/Poster.jpg" alt="Poster" style="width:100%;">
</p>

## 📝 Citation
If you find CADTrack is helpful for your research, please consider citing:

```bibtex
@inproceedings{li2026cadtrack,
  title={CADTrack: Learning Contextual Aggregation with Deformable Alignment for Robust RGBT Tracking},
  author={Li, Hao and Wang, Yuhao and Hu, Xiantao and Hao, Wenning and Zhang, Pingping and Wang, Dong and Lu, Huchuan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={8},
  pages={6109--6117},
  year={2026}
}
```

## 🙏 Acknowledgments
 This repo is based on [STTrack](https://github.com/NJU-PCALab/STTrack) and [IDEA](https://github.com/924973292/IDEA) which are excellent works.
<p align="center"> <b>Star ⭐ this repo if you like our work!</b> </p>
