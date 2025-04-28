<div align="left">
  
# 🔥 DUQ:Dual Uncertainty Quantification for Text-Video Retrieval


## 📣 Updates
* **[2025/01/18]**: We have released the complete training and testing code.
* **[2025/01/21]**: We have released the complete video Q&A code.

## ⚡ Framework

## 😍 Visualization

## 🚀 Quick Start
### Setup

#### Setup code environment
```shell
conda create -n DUQ python=3.8
conda activate DUQ
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Download CLIP Model

Download ViT-B/32 Model: [ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32])

Download ViT-B/16 Model:  [ViT-B/16](https://huggingface.co/openai/clip-vit-base-patch16])

#### Download Datasets

<div align=center>

|  Datasets   |                             Download Link                              |             Training weights              |
|:-----------:|:----------------------------------------------------------------------:|:-----------------------------------------:|
|   MSRVTT    |      [Download](http://ms-multimedia-challenge.com/2017/dataset)       | [Download](https://github.com/OPA067/DUQ) |
|    LSMDC    | [Download](https://sites.google.com/site/describingmovies/download)| [Download](https://github.com/OPA067/DUQ) |
|    MSVD     | [Download](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/) | [Download](https://github.com/OPA067/DUQ) |
| ActivityNet |           [Download](http://activity-net.org/download.html)            | [Download](https://github.com/OPA067/DUQ) |
|  Charades   |         [Download](https://github.com/activitynet/ActivityNet)         | [Download](https://github.com/OPA067/DUQ) |
|   DiDeMo    |       [Download](https://github.com/LisaAnne/LocalizingMoments)        | [Download](https://github.com/OPA067/DUQ) |
|    VATEX    |                              [Download](https://eric-xw.github.io/vatex-website/download.html)  |                     [Download](https://github.com/OPA067/DUQ) |                      | 

</div>

#### 💪 Text-Video Retrieval Training
The training instructions for all datasets are given below, where "--split" needs to be specified according to the dataset size.
##### MSRVTT
```python
python train.py --exp_name=MSRVTT-train --dataset_name=MSRVTT --log_step=100 --evals_per_epoch=5 --batch_size=32 --videos_dir=MSRVTT/videos/ --split=8
```

#### 💪 Example of Text-Video Retrieval Testing
```python
python test.py  --exp_name=MSRVTT-test --save_memory_mode --dataset_name=MSRVTT --batch_size=32 --num_workers=8 --videos_dir=MSRVTT/videos/
```

## 🎗️ Acknowledgments
Our code is based on [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip/), [X-Pool](https://github.com/layer6ai-labs/xpool), [T-Mass](https://github.com/Jiamian-Wang/T-MASS-text-video-retrieval). We sincerely appreciate for their contributions.

