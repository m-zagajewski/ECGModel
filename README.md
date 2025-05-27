# ECG and Clinical Data Mortality Prediction Model

This repository contains a multimodal deep learning model for predicting 30-day mortality risk in ICU patients using both electrocardiogram (ECG) signal data and tabular clinical data.

## Project Overview

The goal of this project is to develop a predictive model that identifies the risk of 30-day mortality for ICU patients by combining:
- ECG time series data (processed with a deep learning model)
- Clinical tabular features (demographics, biomarkers, vital signs, clinical scores)

## Data Structure

### ECG Data
- Format: JSON files (1 file per patient)
- Each file contains 12-lead ECG signal recordings
- Shape: [12 channels, time steps]

### Clinical Tabular Data
- Format: Excel file with one row per patient
- Features include:
  - Demographics: Age, Sex, BMI
  - Biomarkers: IL-6, Procalcitonin, Glucose, Lactate, Base Excess, Anion Gap
  - Vital Signs: Mean Arterial Pressure, PaO2/FiO2 ratio
  - Clinical Scores: SOFA score, Sepsis (binary), Pre-ICU surgery (binary)
  - Outcome Labels: 30-day mortality (binary)

## Model Architecture

The model consists of three main components:

1. **ECG Encoder**: CNN-based or ResNet-based model to extract latent features from multichannel ECG signals.
2. **Tabular Model**: MLP to process clinical variables after normalization and encoding.
3. **Fusion Layer**: Combines outputs from both branches using either simple concatenation or attention-based fusion.

## Project Structure

```
ECGModel/
├── config/                  # Configuration files
│   └── model_config.json    # Sample model configuration
├── src/                     # Source code
│   ├── data/                # Data processing modules
│   │   ├── ecg_processor.py # ECG data loading and preprocessing
│   │   └── tabular_processor.py # Tabular data loading and preprocessing
│   ├── models/              # Model architecture modules
│   │   ├── ecg_encoder.py   # ECG encoder models
│   │   ├── tabular_model.py # Tabular data models
│   │   └── fusion_model.py  # Fusion model architectures
│   ├── train.py             # Training script
│   └── evaluate.py          # Evaluation script
├── data/                    # Data directory (not included in repo)
│   ├── ecg/                 # ECG JSON files
│   └── clinical_data.xlsx   # Tabular clinical data
└── models/                  # Saved models directory (created during training)
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ECGModel.git
cd ECGModel
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Data Preparation

1. Place your ECG JSON files in the `data/ecg/` directory.
2. Place your clinical data Excel file at `data/clinical_data.xlsx`.

### Training

To train the model with the default configuration:

```bash
python src/train.py --config config/model_config.json --device cuda
```

This will perform 5-fold cross-validation and save the trained models in the `models/` directory.

### Evaluation

To evaluate a trained model:

```bash
python src/evaluate.py --config config/model_config.json --model models/model_fold_1.pt --output results --device cuda
```

This will evaluate the model on the test set and save the results in the `results/` directory, including:
- Evaluation metrics (AUROC, F1-score, precision, recall, etc.)
- ROC curve
- Precision-recall curve
- Confusion matrix
- Feature importance visualizations (if using the attention-based fusion model)
- Predictions CSV file

## Configuration

The model configuration is specified in a JSON file. Here's an example:

```json
{
    "ecg_data_dir": "data/ecg",
    "tabular_data_file": "data/clinical_data.xlsx",
    "model_dir": "models",
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "num_epochs": 50,
    "patience": 10,
    "n_splits": 5,
    "random_state": 42,
    "model": {
        "ecg_config": {
            "model_type": "resnet",
            "in_channels": 12,
            "output_dim": 128
        },
        "tabular_config": {
            "model_type": "feature_extractor",
            "input_dim": 20,
            "output_dim": 128,
            "hidden_dims": [256, 128],
            "dropout_rate": 0.3,
            "batch_norm": true
        },
        "fusion_config": {
            "model_type": "attention",
            "fusion_dim": 256,
            "dropout_rate": 0.3
        }
    }
}
```

You can modify this configuration to experiment with different model architectures and hyperparameters.

## Model Options

### ECG Encoder
- `model_type`: "cnn" or "resnet"
- `in_channels`: Number of input channels (ECG leads)
- `output_dim`: Dimension of the output latent representation

### Tabular Model
- `model_type`: "mlp" or "feature_extractor"
- `input_dim`: Dimension of the input features
- `output_dim`: Dimension of the output latent representation
- `hidden_dims`: Dimensions of the hidden layers
- `dropout_rate`: Dropout rate for regularization
- `batch_norm`: Whether to use batch normalization

### Fusion Model
- `model_type`: "concatenation" or "attention"
- `fusion_dim`: Dimension of the fusion layer
- `dropout_rate`: Dropout rate for regularization

## License

This project is licensed under the MIT License - see the LICENSE file for details.