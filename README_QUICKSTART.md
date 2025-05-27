# ECG and Clinical Data Mortality Prediction Model - Quick Start Guide

This quick start guide provides the essential steps to get the model up and running. For more detailed instructions, please refer to the [Comprehensive Usage Guide](USAGE_GUIDE.md).

## Prerequisites

- Python 3.7 or higher
- CUDA-compatible GPU (recommended)
- Required Python packages (install with `pip install -r requirements.txt`)

## Quick Start Steps

### 1. Prepare Your Data

Place your data in the following locations:
- ECG JSON files in `data/ecg/`
- Clinical data Excel file at `data/clinical_data.xlsx`

### 2. Run the Example Script

For a quick demonstration of the full pipeline:

```bash
python run_example.py
```

This script:
- Creates a simplified model configuration
- Trains the model on your data
- Evaluates the model
- Saves results to the `example_output` directory

### 3. Train Your Own Model

To train with the default configuration:

```bash
python src/train.py --config config/model_config.json --device cuda
```

To train with a simpler model (recommended if you encounter 'nan' loss):

```bash
# First, create a simplified configuration
cp config/model_config.json config/simple_model_config.json
```

Edit `config/simple_model_config.json` to use simpler model architectures:
- Change `ecg_config.model_type` to "cnn"
- Change `tabular_config.model_type` to "mlp"
- Change `fusion_config.model_type` to "concatenation"

Then train with:

```bash
python src/train.py --config config/simple_model_config.json --device cuda
```

### 4. Evaluate Your Model

```bash
python src/evaluate.py --config config/model_config.json --model models/model_fold_1.pt --output results --device cuda
```

## Common Issues

### 'nan' Loss Values

If you encounter 'nan' loss values:

1. Use a lower learning rate (e.g., 0.0001)
2. Use a simpler model (see step 3 above)
3. Add gradient clipping to `src/train.py`

### Memory Issues

If you encounter memory issues:

1. Reduce batch size in the configuration file
2. Use a simpler model with smaller dimensions

## Next Steps

For more detailed instructions, including:
- Detailed data preparation steps
- Advanced training options
- Interpreting evaluation results
- Advanced usage scenarios

Please refer to the [Comprehensive Usage Guide](USAGE_GUIDE.md).