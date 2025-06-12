<div align="left">
  
# DUQ:Dual Uncertainty Quantification for Text-Video Retrieval

Our paper ```DUQ:Dual Uncertainty Quantification for Text-Video Retrieval``` has been accepted by IJCAI 2025. In this paper, we propose a novel Dual Uncertainty Quantification (DUQ) model that separately handles uncertainties in intra-pair interaction and inter-pair exclusion. 
Specifically, to enhance intra-pair interaction, we propose an intra-pair similarity uncertainty module to provide similarity-based trustworthy predictions and explicitly model this uncertainty. 
To increase inter-pair exclusion, we propose an inter-pair distance uncertainty module to construct a distance-based diversity probability embeding, thereby widening the gap between similar features. 
The two components work synergistically, jointly improving the calculation of similarity between features.


## 📣 Updates
* **[2025/01/18]**: We have released the complete training and testing code.
* **[2025/01/21]**: We have released the complete video Q&A code.
* **[2025/04/28]**: We have updated some of the details.
* **[2025/04/29]**: ⚡ Our paper has been accepted by IJCAI 2025!
* **[2025/06/01]**: Fixed a bug that had a minor impact on retrival performance.

## 😍 Framework
(1) The Feature Extraction Module maps text and video inputs into a joint embedding space to compute similarity. 
(2) The Intra-pair Similarity Uncertainty Module provides similarity-based trustworthy predictions and explicitly models intra-pair interaction uncertainty. 
(3) The Inter-pair Distance Uncertainty Module constructs distance-based diversity probabilistic embeddings and uses boundary distances to represent inter-pair exclusion differences.
<img src="figures/Framework.png" width="800px" />
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
| ActivityNet |           [Download](http://activity-net.org/download.html)            | [Download](https://github.com/OPA067/DUQ) |
|  Charades   |         [Download](https://github.com/activitynet/ActivityNet)         | [Download](https://github.com/OPA067/DUQ) |
|   DiDeMo    |       [Download](https://github.com/LisaAnne/LocalizingMoments)        | [Download](https://github.com/OPA067/DUQ) |
|    VATEX    |                              [Download](https://eric-xw.github.io/vatex-website/download.html)  |                     [Download](https://github.com/OPA067/DUQ) |                      | 

</div>

#### 💪 Text-Video Retrieval Training
Training instructions for all datasets can be found [here](https://github.com/OPA067/DUQ/script), where you need to specify "--split" according to the size of the dataset.
```python
python train.py --exp_name=MSRVTT-train --dataset_name=MSRVTT --log_step=100 --evals_per_epoch=5 --batch_size=32 --videos_dir=MSRVTT/videos/ --split=8
```

#### 💪 Example of Text-Video Retrieval Testing
```python
python test.py --exp_name=MSRVTT-test --dataset_name=MSRVTT --batch_size=32 --videos_dir=MSRVTT/videos/
```

## 🎗️ Acknowledgments
Our code is based on [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip/), [X-Pool](https://github.com/layer6ai-labs/xpool). We sincerely appreciate for their contributions.

