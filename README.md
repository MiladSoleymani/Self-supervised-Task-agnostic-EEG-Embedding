# Self-supervised Task-agnostic EEG Embedding

Self-supervised learning of EEG embeddings that can be reused across tasks and datasets without task labels. Inspired by PASE (Problem-Agnostic Speech Encoder): a shared encoder is trained so that several small "worker" networks can predict handcrafted signal features from its embedding. The embedding is then evaluated on downstream classification.

## Method

- **Encoder:** per-channel frequency-domain features are placed on a 2-D scalp grid by electrode position (10-20 system), and a 2-D CNN maps them to a 256-d embedding.
- **Workers (self-supervised targets):** wavelet transform energy, wavelet packet energy, power spectral density, kurtosis, skewness and peak-to-peak amplitude. They are configured in `configs/pase_base/workers.json`.
- **Downstream:** a classifier, or EEGNet as a baseline, trained on top of the embedding, with optional k-fold cross-validation.

## Datasets

BCI Competition IV-2a (motor imagery), CHB-MIT (seizure detection), the Lee et al. dataset, and a clinical dataset (Klinik). Preparation scripts are in `scripts/tools/`.

## Installation

```bash
git clone https://github.com/MiladSoleymani/Self-supervised-Task-agnostic-EEG-Embedding.git
cd Self-supervised-Task-agnostic-EEG-Embedding
pip install torch "pytorch-lightning==1.6.*" "jsonargparse[signatures]" torchmetrics hydra-core \
            mne mne-features PyWavelets scipy scikit-learn pandas
```

The code uses the PyTorch Lightning 1.x `LightningCLI` API.

## Usage

```bash
# Self-supervised pre-training on BCI IV-2a
PYTHONPATH=./ python pase_eeg/cli/train.py fit --config=configs/pase_base/config_BCIIV2a.yaml

# Downstream classifier on BCI IV-2a
PYTHONPATH=./ python pase_eeg/cli/train.py fit --config=configs/simple_classifier/config_BCIIV2a.yaml

# k-fold cross-validation
PYTHONPATH=./ python pase_eeg/cli/crossval.py \
    --config=configs/simple_classifier/config_trainer_base.yaml \
    --config=configs/simple_classifier/crossval/config_BCIIV2a.yaml
```

Data paths and trainer settings are set in the YAML files under `configs/`. The configs use a `CodeSnapshot` callback from `pytorch-lightning-snippets`; add that repo to `PYTHONPATH` or remove the callback.

## Project structure

```
pase_eeg/
  nn/            encoder, workers and classifiers
  data/          dataset loaders and feature transforms
  lit_modules/   Lightning modules and data modules
  cli/           training and cross-validation entry points
configs/         experiment configs and EEG channel layouts
scripts/tools/   dataset download and preprocessing
notebooks/       data exploration and EEGNet baseline
```
