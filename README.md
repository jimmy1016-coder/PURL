# PURL for Noisy Correspondence SV (Pre-release)

This repository contains code for:

**"PURL: Pairwise Unlearning with Reliability Learning for Noisy Correspondence in Supervised Contrastive Speaker Verification"**  
Anonymous submission to Interspeech 2026.

Repository name is currently legacy (`chns_4090`), but the method implemented here follows the PURL formulation described in the paper draft.

## Project Status

This repository is a **pre-release research codebase**.

- It is being prepared for public release and reproducibility.
- The corresponding Interspeech 2026 submission is **under review** (not yet accepted).
- APIs, configs, and default paths may still change.

## Method Overview (Paper-aligned)

PURL is a post-hoc, pair-level unlearning framework for supervised contrastive speaker verification under noisy correspondence.

- **Problem**: Positive pairs can be mismatched in large-scale data, which corrupts pairwise supervision.
- **Stage 1 (reliability estimation)**: Fit a 2-component GMM on pairwise cosine similarities in a pretrained embedding space and compute pair confidence `w_ij`.
- **Stage 2 (uncertainty-guided update)**:
  - retain reliable pairs with SupCon objective
  - suppress unreliable pairs with weighted cosine repulsion proportional to `(1 - w_ij)`
- **Goal**: reduce harmful correspondences while preserving clean intra-speaker compactness.

In this codebase, this is implemented through `NCUTrainer` with `ncu_loss_type: soft`.

## What Is Included

This public pre-release keeps four training configs:

- `configs/supcon.yaml`
- `configs/supcon_resnet34.yaml`
- `configs/ncu_30ep.yaml`
- `configs/ncu_30ep_resnet34.yaml`

All machine-specific absolute paths were replaced with placeholders. Update them before running.

## Experiments in This Release

The retained configs correspond to the two backbones and post-hoc setup used in the paper:

- **ECAPA-TDNN**
  - base SupCon pretraining: `configs/supcon.yaml`
  - 30-epoch post-hoc update (PURL/NCU): `configs/ncu_30ep.yaml`
- **Thin ResNet-34**
  - base SupCon pretraining: `configs/supcon_resnet34.yaml`
  - 30-epoch post-hoc update (PURL/NCU): `configs/ncu_30ep_resnet34.yaml`

## Environment Setup

Python 3.10 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

If you use conda:

```bash
conda create -n chns python=3.10 -y
conda activate chns
pip install -r requirements.txt
pip install -e .
```

## Important Runtime Notes

1) Configs use class paths like `trainers.*`, so run with:

```bash
PYTHONPATH=src python run_train.py ...
PYTHONPATH=src python run_test.py ...
```

2) NCU/PURL post-hoc runs assume a pretrained SupCon checkpoint via `--ckpt_path`.

## Dataset Preparation

1. Download VoxCeleb data from the official source: [VoxCeleb download page](https://mm.kaist.ac.kr/datasets/voxceleb/).
2. Convert source audio to the format expected by your experiments (commonly 16 kHz mono wav/flac).
3. Create `spk2utt` mapping files:

```bash
./scripts/make_spk2utt_and_utt2spk.sh /path/to/vox2_dev_wav/wav
```

4. Edit each config to set placeholders such as:

- `/path/to/vox2_dev_wav/wav`
- `/path/to/vox2_dev_wav/wav/spk2utt`
- `/path/to/vox1_test_wav/wav`
- `/path/to/voxceleb1_H.txt`

5. Ensure `spk2utt` files are consistent with your data layout and path conventions.

## Train

SupCon (ECAPA):

```bash
PYTHONPATH=src python run_train.py --config configs/supcon.yaml
```

SupCon (ResNet34):

```bash
PYTHONPATH=src python run_train.py --config configs/supcon_resnet34.yaml
```

PURL post-hoc update (ECAPA, 30 epochs):

```bash
PYTHONPATH=src python run_train.py --config configs/ncu_30ep.yaml --ckpt_path /path/to/base_supcon.ckpt
```

PURL post-hoc update (ResNet34, 30 epochs):

```bash
PYTHONPATH=src python run_train.py --config configs/ncu_30ep_resnet34.yaml --ckpt_path /path/to/base_supcon_resnet34.ckpt
```

## Evaluation

```bash
PYTHONPATH=src python run_test.py --config configs/ncu_30ep.yaml --ckpt_name epoch=XXX_step=YYYY
```

or

```bash
PYTHONPATH=src python run_test.py --config configs/supcon.yaml --ckpt_name epoch=XXX_step=YYYY
```

`--ckpt_name` should be the checkpoint filename stem under `<save_dir>/checkpoints`.

## Reproducibility Notes

- `run_train.py` supports `--ckpt_path` for explicit resume.
- `run_test.py` expects `--ckpt_name` and uses the checkpoint directory from logger config.
- For smoke tests with `limit_test_batches` or `fast_dev_run`, only partial trial embeddings are computed.
- `ncu_30ep*.yaml` currently uses `voxceleb1_H`-style placeholder for default test trials.

## Planned Updates

- Add final camera-ready paper link and BibTeX after decision/publication.
- Add exact benchmark scripts and standardized evaluation table templates for easier replication.

## Citation

If you use this code, please cite the final paper once available.

Temporary reference:

> PURL: Pairwise Unlearning with Reliability Learning for Noisy Correspondence in Supervised Contrastive Speaker Verification (Interspeech 2026 submission, under review)

