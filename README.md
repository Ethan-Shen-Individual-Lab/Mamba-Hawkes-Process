# Adaptive Mamba Hawkes Process (A-MHP)

Code for our WWW 2026 paper: [Mamba Hawkes Process for Event Sequence Modeling](https://dl.acm.org/doi/abs/10.1145/3774904.3792583).

## Overview

This repository implements A-MHP and MHP. This is an **initial release** intended for research reproduction. The current codebase includes a vendored copy of the **Mamba** implementation (and a Transformer module).

Both models are available under a single entry point, selectable via `--model {amhp,mhp}`.

## Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (recommended)

```bash
pip install torch numpy einops tqdm
```

## Data Preparation

Download datasets from the [Neural Hawkes Process repository](https://drive.google.com/drive/folders/0BwqmV0EcoUc8UklIR1BKV25YR1U?resourcekey=0-OrlU87jyc1m-dVMmY5aC4w).

Place the downloaded folders in a directory named `data/`:
```
<repo_root>/
├── data/
│   ├── data_bookorder/
│   ├── data_so/
│   ├── data_hawkes/
│   ├── data_retweet/
│   └── data_mimic/
├── Main.py
├── ablation.py
├── Utils.py
├── preprocess/
│   └── Dataset.py
└── transformer/
    ├── Models.py
    ├── Models_mhp.py
    └── mambapy/
        ├── mamba.py
        └── mamba_mhp.py
```

## Running Experiments

### Main Experiments (A-MHP)

Run all datasets with default loss weights (β=1.0, γ=1e-4):
```bash
python Main.py
```

Run specific dataset:
```bash
python Main.py --dataset SO
```

Specify custom loss weights:
```bash
python Main.py --dataset Financial --beta 1.0 --gamma 1e-4
```

Specify cross-validation fold (for Financial, SO, Mimic datasets):
```bash
python Main.py --dataset SO --fold 1
python Main.py --dataset Financial --fold 3 --beta 1.0 --gamma 1e-4
```

### Baseline Experiments (MHP)

Run all datasets with the baseline MHP model:
```bash
python Main.py --model mhp
```

Run specific dataset:
```bash
python Main.py --model mhp --dataset SO
python Main.py --model mhp --dataset Financial --fold 2
```

### Ablation Study

Test individual mechanisms:
```bash
python ablation.py --dataset SO --beta 1.0 --gamma 1e-4
```

This runs two variants:
- `time_scaling_only`: GRU time-scaling without dual-channel
- `dual_channel_only`: Dual-channel without GRU time-scaling

Ablation study also supports fold parameter:
```bash
python ablation.py --dataset SO --fold 2 --beta 1.0 --gamma 1e-4
```

## Model Architecture

Key hyperparameters (all datasets use n_layers=4):

| Parameter | Value | Description |
|-----------|-------|-------------|
| d_state | 16 | State space dimension |
| d_inner | 2×d_model | Internal feature dimension (auto-computed) |
| d_conv | 4 | Convolution kernel size |
| expand_factor | 2 | Feature expansion ratio |
| dt_rank | ⌈d_model/16⌉ | Time-step projection rank |

Dataset-specific d_model and learning rates are specified in paper Table.

## Results

Results are saved in log files:
- `log_{dataset}_A-MHP_Mamba_pure_OOD.txt`: A-MHP test metrics per epoch
- `log_{dataset}_A-MHP_Mamba_pure_OOD_train.txt`: A-MHP training metrics per epoch
- `log_{dataset}_MHP_Mamba_pure_OOD.txt`: MHP test metrics per epoch

For ablation experiments:
- `log_{dataset}_Ablation_{variant}_OOD.txt` and corresponding `_train.txt` files

## Implementation Notes

- Xavier normal initialization with layer-specific gains (0.8 for embeddings, 0.6 for predictors)
- Adam optimizer with β₁=0.9, β₂=0.95, weight decay 1e-5
- Gradient clipping (max norm 1.0)

Note: The Retweet dataset is configured with lr=1e-2 as specified in the paper, but during execution, the code automatically adjusts it to 1e-3 (0.1× reduction, which matches the config set in paper table) for numerical stability. This is a necessary engineering adjustment to prevent NaN errors during training.

## Citation

If you use this code, please cite our paper by:

```bibtex
@inproceedings{dai2026mhp,
  author = {Dai, Shan and Shen, Yuyang and Liang, Yuyang and Ma, Chenhao and Gao, Anningzhe},
  title = {Mamba Hawkes Process for Event Sequence Modeling},
  year = {2026},
  isbn = {9798400723070},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  doi = {10.1145/3774904.3792583},
  booktitle = {Proceedings of the ACM Web Conference 2026},
  pages = {7464–7473},
  numpages = {10},
  keywords = {asynchronous event, hawkes process, state space model, time-varying state, long-term dependency},
  location = {United Arab Emirates},
  series = {WWW '26}
}
```

## License

This code is released under the MIT License.
