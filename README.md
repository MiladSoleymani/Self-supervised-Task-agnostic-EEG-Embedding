# Self-supervised Task-agnostic EEG Embedding

A PyTorch Lightning implementation of self-supervised learning for EEG signal embeddings that generalize across multiple tasks and datasets without requiring task-specific labels.

## Key Features

- **Self-supervised Learning**: Learn meaningful EEG representations without labeled data
- **Task-agnostic Embeddings**: Transfer learned features to various downstream tasks (classification, detection, etc.)
- **Multiple Dataset Support**: Pre-configured for BCI Competition IV-2a, LEE, CHB-MIT, and Klinik datasets
- **Flexible Architecture**: Modular design with customizable encoder networks
- **PyTorch Lightning Integration**: Built-in distributed training, logging, and checkpointing
- **Cross-validation Support**: Automated k-fold cross-validation for robust evaluation

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)

### Setup
```bash
# Clone the repository
git clone https://github.com/MiladSoleymani/Self-supervised-Task-agnostic-EEG-Embedding.git
cd Self-supervised-Task-agnostic-EEG-Embedding

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pytorch-lightning torch torchvision
pip install numpy scipy scikit-learn
pip install mne  # For EEG data processing
```

## Quick Start

### Training a Self-supervised Model

```bash
# Train PASE model on synthetic data
PYTHONPATH=./ python pase_eeg/cli/train.py fit --model=PaseLit --data=EEGSynthetichDataLit

# Train on specific dataset with custom config
PYTHONPATH=./ python pase_eeg/cli/train.py fit --config=configs/pase_base/config_BCIIV2a.yaml --trainer.gpus=1
```

### Fine-tuning on Downstream Tasks

```bash
# Train classifier on BCI Competition IV-2a
PYTHONPATH=./ python pase_eeg/cli/train.py fit --config=configs/simple_classifier/config_BCIIV2a.yaml

# Run cross-validation
PYTHONPATH=./ python pase_eeg/cli/crossval.py --config=configs/simple_classifier/crossval/config_BCIIV2a.yaml
```

## Project Structure

```
├── configs/                      # Configuration files
│   ├── pase_base/               # Self-supervised model configs
│   ├── simple_classifier/       # Downstream task configs
│   └── eeg_recording_standard/  # EEG channel configurations
├── pase_eeg/                    # Main package
│   ├── nn/                      # Neural network modules
│   │   ├── models/              # Model architectures
│   │   └── modules/             # Building blocks and layers
│   ├── data/                    # Dataset implementations
│   │   ├── bci/                 # BCI Competition datasets
│   │   ├── chb_mit/             # CHB-MIT seizure dataset
│   │   ├── LEE/                 # LEE dataset
│   │   └── Klinik/              # Clinical EEG dataset
│   ├── lit_modules/             # PyTorch Lightning modules
│   └── cli/                     # Command-line interfaces
├── notebooks/                   # Jupyter notebooks
│   ├── BCI_Competition_IV.ipynb # BCI dataset exploration
│   └── EEGNet/                  # EEGNet implementation
└── scripts/                     # Data preparation scripts
    └── tools/                   # Dataset downloaders and preprocessors
```

## Supported Datasets

### BCI Competition IV-2a
Motor imagery dataset with 9 subjects, 4 classes
```bash
python scripts/tools/prepare_EEG_data.py --dataset bci_iv_2a
```

### CHB-MIT
Pediatric seizure detection dataset
```bash
python scripts/tools/prepare_chb_data.py
```

### LEE Dataset
EEG dataset for various cognitive tasks
```bash
python scripts/tools/prepare_LEE_data.py
```

### Custom Datasets
Implement your dataset by extending `pase_eeg.data.base.BaseEEGDataset`

## Model Architecture

The self-supervised model uses a multi-task learning approach with:
- **Encoder**: Extracts task-agnostic features from raw EEG signals
- **Minion Networks**: Multiple auxiliary tasks for self-supervision
  - Temporal context prediction
  - Channel reconstruction
  - Frequency domain tasks
- **Embedder**: Projects features to fixed-dimensional embeddings

## Configuration

All experiments are configured through YAML files:

```yaml
# Example config structure
model:
  class_path: pase_eeg.lit_modules.pase_lit.PaseLit
  init_args:
    encoder_cfg:
      emb_dim: 256
    minion_cfg:
      - name: temporal_context
        weight: 1.0

data:
  class_path: pase_eeg.data.bci.BCIDataModule
  init_args:
    batch_size: 32
    num_workers: 4
```

## Evaluation

### Embedding Quality Metrics
- Linear evaluation protocol
- t-SNE visualization
- Cross-dataset transfer learning

### Downstream Task Performance
- Classification accuracy
- F1 scores
- Cross-validation results

## Notebooks

Interactive notebooks for exploration and visualization:
- `BCI_Competition_IV.ipynb`: Dataset analysis and preprocessing
- `dnn_architecture.ipynb`: Model architecture visualization
- `EEGNet-PyTorch.ipynb`: Baseline model implementation

## Citation

If you use this code in your research, please cite:
```bibtex
@misc{soleymani2024selfsupervised,
  title={Self-supervised Task-agnostic EEG Embedding},
  author={Soleymani, Milad},
  year={2024},
  publisher={GitHub},
  url={https://github.com/MiladSoleymani/Self-supervised-Task-agnostic-EEG-Embedding}
}
```

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PyTorch Lightning team for the excellent framework
- MNE-Python for EEG data processing utilities
- Authors of the original PASE paper for inspiration
