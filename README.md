<div align="center">

# IJCAI2025, GuangZhou
# Dual Uncertainty Quantification for Text-Video Retrieval

**Accepted by IJCAI 2025** 🎉

[Paper](https://www.ijcai.org/proceedings/2025/643) | [Code](https://github.com/OPA067/DUQ)

</div>

## 📝 Introduction

We propose a novel **Dual Uncertainty Quantification (DUQ)** model that separately handles two types of uncertainty in text-video retrieval:

- **Intra-pair similarity uncertainty**: Provides similarity-based trustworthy predictions and explicitly models the uncertainty within each text-video pair.
- **Inter-pair distance uncertainty**: Constructs a distance-based diversity probability embedding, widening the gap between similar features across different pairs.

The two components work synergistically, jointly improving the accuracy of cross-modal similarity computation.

## 📣 Updates

| Date | Description |
|------|-------------|
| 2025/01/18 | Released complete training and testing code |
| 2025/04/29 | Paper accepted by IJCAI 2025 |
| 2025/06/01 | Updated code details and documentation |

## 📁 Structure

```
DUQ/
├── main_retrieval.py          # Main entry point for training & evaluation
├── requirements.txt           # Python dependencies
├── models/
│   ├── modeling.py            # Core DUQ model
│   ├── module_clip.py         # CLIP backbone (visual & textual encoders)
│   ├── module_prob.py         # Probabilistic embedding module (inter-pair uncertainty)
│   ├── module_edl.py          # Evidential deep learning module (intra-pair uncertainty)
│   ├── module_cross.py        # Cross-modal Transformer
│   ├── module_transformer.py  # Temporal Transformer
│   ├── optimization.py        # Optimizer & learning rate scheduler
│   ├── tokenization_clip.py   # Text tokenizer for CLIP
│   ├── until_config.py        # Model configuration utilities
│   ├── until_module.py        # Common model utilities
│   ├── file_utils.py          # File I/O helpers
│   ├── cross-base/
│   │   └── cross_config.json  # Cross-modal Transformer config
│   └── bpe_simple_vocab_16e6.txt.gz  # BPE vocabulary for tokenization
├── dataloaders/
│   ├── __init__.py
│   ├── data_dataloaders.py    # Data loader factory & collate functions
│   ├── dataloader_msrvtt_retrieval.py  # MSRVTT dataset loader
│   ├── dataloader_didemo_retrieval.py  # DiDeMo dataset loader
│   ├── dataloader_charades_retrieval.py  # Charades dataset loader
│   ├── dataloader_retrieval.py  # Base retrieval dataset class
│   ├── rawvideo_util.py       # Raw video reading utilities
│   ├── video_transforms.py    # Video augmentation transforms
│   ├── random_erasing.py      # Random erasing augmentation
│   ├── rand_augment.py        # RandAugment policy
│   └── functional.py          # Functional transform helpers
├── utils/
│   ├── __init__.py
│   ├── metrics.py             # Retrieval evaluation metrics (R@K, MdR, MnR)
│   ├── metrics_qa.py          # QA-specific evaluation metrics
│   ├── logger.py              # Logging utilities
│   ├── metric_logger.py       # Metric logging & smoothing
│   ├── util.py                # General utility functions
│   └── comm.py                # Distributed communication helpers
├── script/
│   ├── run_MSRVTT.sh          # Training & eval script for MSRVTT
│   ├── run_DiDeMo.sh          # Training & eval script for DiDeMo
│   ├── run_Charades.sh        # Training & eval script for Charades
│   └── run_test.sh            # Quick test script
├── msrvtt/                    # MSRVTT dataset metadata
│   ├── MSRVTT_data.json       # Video-id to caption mappings
│   ├── MSRVTT_train.9000.csv  # 9K train split
│   ├── MSRVTT_train.7000.csv  # 7K train split
│   └── MSRVTT_test.1000.csv   # 1K test split
├── preprocess/
│   └── compress_video.py      # Video preprocessing & compression
├── docs/                      # Paper & supplementary materials
│   ├── DUQ_Main.pdf
│   ├── DUQ_Poster.pdf
│   └── DUQ_Author_Response.pdf
├── experiments/               # Output directory for logs & checkpoints
│   └── MSRVTT/                # Per-dataset experiment outputs
│       └── <timestamp>/       # Timestamped run directories
│           ├── log.txt        # Training & evaluation log
│           └── pytorch_model.bin.*  # Model checkpoints
└── figures/                   # Motivation & framework diagrams
    ├── Framework.png
    ├── Framework.pdf
    ├── Motivation.png
    └── Motivation.pdf
```

## 😍 Visualization

### Motivation

<p float="left">
  <img src="figures/Motivation.png" width="80%" />
</p>

### Framework

<p float="left">
  <img src="figures/Framework.png" width="80%" />
</p>

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n DUQ python=3.9
conda activate DUQ

# Install dependencies
pip install -r requirements.txt
pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
```

### 2. Download CLIP Model

Place the pretrained CLIP model in the `models/` directory:

```bash
cd DUQ/models

# ViT-B/32 (default)
wget https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt

# ViT-B/16 (optional)
# wget https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt

# ViT-L/14 (optional)
# wget https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
```

### 3. Download Datasets

| Dataset | Download Link | Description |
|---------|---------------|-------------|
| MSRVTT | [Download](http://ms-multimedia-challenge.com/2017/dataset) | 10K YouTube videos with 200K captions |
| LSMDC | [Download](https://sites.google.com/site/describingmovies/download) | 118K video clips from 202 movies |
| ActivityNet | [Download](http://activity-net.org/download.html) | 20K videos with 100K temporal annotations |
| Charades | [Download](https://github.com/activitynet/ActivityNet) | 10K daily activity videos with 27K captions |
| DiDeMo | [Download](https://github.com/LisaAnne/LocalizingMoments) | 10K videos with 42K localization annotations |
| VATEX | [Download](https://eric-xw.github.io/vatex-website/download.html) | 41K videos with 825K bilingual captions |

### 4. Training

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --master_port 2501 \
    --nproc_per_node=1 \
    main_retrieval.py \
    --do_train 1 \
    --workers 8 \
    --n_display 100 \
    --epochs 5 \
    --lr 1e-4 \
    --coef_lr 1e-3 \
    --batch_size 32 \
    --batch_size_val 32 \
    --anno_path MSRVTT \
    --video_path MSRVTT/videos \
    --datatype msrvtt \
    --max_words 32 \
    --max_frames 12 \
    --video_framerate 1 \
    --output_dir experiments/MSRVTT

# Multi-GPU (e.g., 4 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --master_port 2501 \
    --nproc_per_node=4 \
    main_retrieval.py \
    --do_train 1 \
    --workers 8 \
    --n_display 100 \
    --epochs 5 \
    --lr 1e-4 \
    --coef_lr 1e-3 \
    --batch_size 128 \
    --batch_size_val 128 \
    --anno_path MSRVTT \
    --video_path MSRVTT/videos \
    --datatype msrvtt \
    --max_words 32 \
    --max_frames 12 \
    --video_framerate 1 \
    --output_dir experiments/MSRVTT
```

### 5. Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --master_port 2502 \
    --nproc_per_node=1 \
    main_retrieval.py \
    --do_eval 1 \
    --workers 8 \
    --n_display 100 \
    --batch_size 32 \
    --batch_size_val 32 \
    --anno_path MSRVTT \
    --video_path MSRVTT/videos \
    --datatype msrvtt \
    --max_words 32 \
    --max_frames 12 \
    --video_framerate 1 \
    --output_dir experiments/MSRVTT \
    --init_model experiments/MSRVTT/<RUN_NAME>/pytorch_model.bin.<EPOCH>
```

> 📌 For more details, refer to the [`script/`](https://github.com/OPA067/DUQ/tree/master/script) directory.

## 💪 Experiments

Training logs and checkpoints are saved under `experiments/<dataset>/<timestamp>/`.

### Log File Format

Each experiment generates a `log.txt` file containing the following sections:

**1. Configuration Parameters**
```
[2026-09-03 10:42:53 Model 170 INFO]: Effective parameters:
  <<< agg_module: seqTransf
  <<< alpha: 0.1
  <<< anno_path: ./data/MSRVTT
  <<< base_encoder: ViT-B/32
  <<< batch_size: 32
  <<< beta: 0.01
  <<< coef_lr: 0.001
  <<< data_path: ./data
  <<< datatype: msrvtt
  <<< device: cuda
  <<< distributed: True
  <<< do_eval: False
  <<< do_train: True
  <<< epochs: 5
  <<< feature_framerate: 1
  <<< gamma: 1.0
  <<< init_model: None
  <<< interaction: 0
  <<< local_rank: 0
  <<< lr: 0.0001
  <<< max_frames: 12
  <<< max_words: 24
  <<< n_display: 50
  <<< num_hidden_layers: 4
  <<< output_dir: None
  <<< seed: 42
  <<< split_batch: 2
  <<< video_framerate: 1
  <<< video_path: None
  <<< warmup_proportion: 0.1
  <<< weight_decay: 0.01
  <<< workers: 5
  <<< world_size: 1
```

**2. Model Statistics**
```
Total params: 185.48M
Trainable params: 183.12M
```

**3. Zero-shot Evaluation (Before Training)**
```
T->V: R@1: 31.6 - R@5: 56.4 - R@10: 66.3 - R@Sum: 154.3 - MdR: 4.0 - MnR: 30.1
V->T: R@1: 33.2 - R@5: 57.5 - R@10: 66.4 - R@Sum: 157.1 - MdR: 4.0 - MnR: 27.7
```

**4. Running Testing/Training Info**
```
Running testing: Num examples = 1000, Batch size = 2, Num steps = 500
Running training: Num examples = 6513, Batch size = 32, Num steps = 5625
```

**5. Training Progress (per 50 iterations)**
```
[2026-09-03 11:16:13 Model 500 INFO]: eta: 3:50:02, epoch: 1/5, iter: 50/5625, time: 4.0233, data: 0.0486, loss: 3.5292, lr: 0.000000442/0.000100000, logit: 4.6061, memory: 19814, grad_norm: 0.0181
```

| Field | Description | Example |
|-------|-------------|---------|
| `eta` | Estimated time remaining (HH:MM:SS) | `4:19:13` |
| `epoch` | Current epoch / total epochs | `1/5` |
| `iter` | Current iter / total iters | `4300/5625` |
| `time` | Time per iteration (seconds) | `4.0233` |
| `data` | Data loading time (seconds) | `0.0486` |
| `loss` | Training loss value | `0.6159` |
| `lr` | Learning rate (current/max with warmup) | `0.000000094/0.000100000` |
| `logit` | Logit scale value (temperature) | `4.6061` |
| `memory` | GPU memory usage (MB) | `19814` |
| `grad_norm` | Gradient norm (for monitoring) | `0.0181` |

**6. Evaluation Results (after each epoch)**
```
[2026-09-03 15:23:08 Model 2837 INFO]: Evaluation Results:
Text-to-Video Retrieval:
R@1: 42.6, R@5: 70.2, R@10: 80.3, R@Sum: 193.1, MdR: 2.0, MnR: 16.7
Video-to-Text Retrieval:
R@1: 46.5, R@5: 73.3, R@10: 82.3, R@Sum: 202.1, MdR: 2.0, MnR: 12.1
```

| Metric | Description |
|--------|-------------|
| R@K | Recall at K (1, 5, 10) - higher is better |
| R@Sum | Sum of R@1 + R@5 + R@10 - combined retrieval performance |
| MdR | Median Rank - lower is better (position of first relevant result) |
| MnR | Mean Rank - lower is better (average position of relevant results) |

**7. Model Saving and Best Model Tracking**
```
[2026-09-03 15:23:08 Model 2842 INFO]: Saving current best model...
Best Text-to-Video Retrieval: R1: 42.6
Best Video-to-Text Retrieval: R1: 46.5
```

**8. Final Evaluation (Best Model)**
```
[2026-09-03 15:23:20 Model 2858 INFO]: Final Evaluation of the best model:
Text-to-Video Retrieval:
R@1: 42.6, R@5: 70.2, R@10: 80.3, R@Sum: 193.1, MdR: 2.0, MnR: 16.7
Video-to-Text Retrieval:
R@1: 46.5, R@5: 73.3, R@10: 82.3, R@Sum: 202.1, MdR: 2.0, MnR: 12.1
```

**9. Training Summary**
```
Total Training Time: 05h 06min 23s
```

### Example Results (MSRVTT)

| Metric | Text→Video | Video→Text |
|--------|------------|------------|
| R@1 | 31.6 | 33.2 |
| R@5 | 56.4 | 57.5 |
| R@10 | 66.3 | 66.4 |
| R@Sum | 154.3 | 157.1 |
| MdR | 4.0 | 4.0 |
| MnR | 30.1 | 27.7 |

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--do_train` | Enable training mode | 0 |
| `--do_eval` | Enable evaluation mode | 0 |
| `--datatype` | Dataset name | msrvtt |
| `--max_words` | Maximum text token length | 24 |
| `--max_frames` | Maximum video frames | 12 |
| `--lr` | Learning rate | 1e-4 |
| `--coef_lr` | CLIP branch LR coefficient | 1e-3 |
| `--epochs` | Training epochs | 5 |
| `--batch_size` | Batch size | 32 |
| `--init_model` | Pretrained checkpoint path | - |
| `--output_dir` | Output directory | experiments/ |

## 🎗️ Acknowledgments

Our code is built upon [Clip4clip](https://github.com/ArrowLuo/CLIP4Clip/), [X-Pool](https://github.com/layer6ai-labs/xpool), and [HBI](https://github.com/jpthu17/HBI/tree/main). We sincerely appreciate their contributions.

## 📚 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{Liu2025DUQ,
  author    = {Xin Liu and Shibai Yin and Jun Wang and Jiaxin Zhu and Xingyang Wang and Yee-Hong Yang},
  title     = {DUQ: Dual Uncertainty Quantification for Text-Video Retrieval},
  booktitle = {Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence},
  year      = {2025},
  pages     = {5779--5787},
  doi       = {10.24963/ijcai.2025/643}
}
```

## 📬 Contact

If you have any questions, feel free to reach out:

- **Issues**: For bug reports, feature requests, or general questions, please open a [GitHub Issue](https://github.com/OPA067/DUQ/issues).
- **Email**: `xinl067@163.com`

We welcome contributions and suggestions!

## 📄 License

This project is released for academic research use only.
