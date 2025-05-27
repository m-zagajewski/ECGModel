   # Comprehensive Guide to Running the ECG and Clinical Data Mortality Prediction Model

This guide provides detailed instructions on how to properly run the ECG and Clinical Data Mortality Prediction Model. It covers data preparation, training, evaluation, and troubleshooting common issues.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Data Preparation](#data-preparation)
3. [Training the Model](#training-the-model)
4. [Evaluating the Model](#evaluating-the-model)
5. [Troubleshooting](#troubleshooting)
6. [Advanced Usage](#advanced-usage)

## Prerequisites

Before you begin, make sure you have:

1. Python 3.7 or higher installed
2. Git installed (for cloning the repository)
3. CUDA-compatible GPU (recommended for faster training)

## Data Preparation

### ECG Data

1. Create a directory for ECG data if it doesn't exist:
   ```bash
   mkdir -p data/ecg
   ```

2. Place your ECG JSON files in the `data/ecg/` directory. Each JSON file should:
   - Be named with the patient ID (e.g., `123_2022.json` for patient ID "123/2022")
   - Contain a `leads` array with 12 lead objects
   - Each lead object should have a `signal` array containing the ECG signal values

   Example ECG JSON structure:
   ```json
   {
     "leads": [
       {
         "lead_name": "I",
         "signal": [0.1, 0.2, 0.3, ...]
       },
       ...
     ]
   }
   ```

3. Verify your ECG data:
   ```bash
   python check_ecg.py
   ```

### Clinical Data

1. Prepare your clinical data Excel file and place it at `data/clinical_data.xlsx`. The file should:
   - Have one row per patient
   - Include a column named `KG` with patient IDs that match the ECG filenames (but with "/" instead of "_")
   - Include a column named `ZGON wewnątrzszpitalnie` with values "TAK" (Yes) or "NIE" (No) for mortality
   - Include other clinical features like `PLEC` (gender), `WIEK` (age), etc.

2. Verify your clinical data:
   ```bash
   python check_data.py
   ```

## Training the Model

### Basic Training

To train the model with the default configuration:

```bash
python src/train.py --config config/model_config.json --device cuda
```

This will:
1. Load the ECG and clinical data
2. Perform 5-fold cross-validation
3. Train the model for each fold
4. Save the trained models in the `models/` directory
5. Save cross-validation results in `models/cv_results.json`

### Training with a Simpler Model

If you encounter numerical issues (like 'nan' loss), try training with a simpler model:

1. Create a simplified configuration file:
   ```bash
   cp config/model_config.json config/simple_model_config.json
   ```

2. Edit `config/simple_model_config.json` to use simpler model architectures:
   - Change `ecg_config.model_type` to "cnn" instead of "resnet"
   - Change `tabular_config.model_type` to "mlp" instead of "feature_extractor"
   - Change `fusion_config.model_type` to "concatenation" instead of "attention"
   - Reduce the output dimensions (e.g., change `output_dim` to 64 instead of 128)

3. Train with the simplified configuration:
   ```bash
   python src/train.py --config config/simple_model_config.json --device cuda
   ```

### Training with CPU Only

If you don't have a GPU, you can train on CPU (though it will be much slower):

```bash
python src/train.py --config config/model_config.json --device cpu
```

### Monitoring Training Progress

The training script will print progress information for each epoch, including:
- Training loss
- Validation loss
- Validation AUROC
- Validation F1 score

## Evaluating the Model

### Basic Evaluation

To evaluate a trained model:

```bash
python src/evaluate.py --config config/model_config.json --model models/model_fold_1.pt --output results --device cuda
```

This will:
1. Load the trained model
2. Evaluate it on the test set
3. Save evaluation results in the `results/` directory

### Interpreting Evaluation Results

The evaluation script generates several outputs:

1. `metrics.json`: Contains evaluation metrics including:
   - AUROC (Area Under the Receiver Operating Characteristic curve)
   - F1 score
   - Precision
   - Recall
   - Specificity
   - Accuracy

2. Visualization plots:
   - `roc_curve.png`: ROC curve showing the trade-off between true positive rate and false positive rate
   - `precision_recall_curve.png`: Precision-recall curve
   - `confusion_matrix.png`: Confusion matrix showing true positives, false positives, true negatives, and false negatives

3. Feature importance visualizations (if using the attention-based fusion model):
   - `tabular_feature_importance.png`: Importance of each tabular feature
   - `modality_importance.png`: Relative importance of ECG vs. tabular data

4. `predictions.csv`: CSV file containing:
   - Patient IDs
   - True labels
   - Predicted probabilities
   - Predicted binary labels

## Troubleshooting

### 'nan' Loss Values

If you encounter 'nan' (not a number) loss values during training:

1. **Reduce the learning rate**: Edit the configuration file to use a smaller learning rate (e.g., 0.0001 instead of 0.001)

2. **Use gradient clipping**: Add the following code to the `train_epoch` function in `src/train.py`:
   ```python
   # After loss.backward() and before optimizer.step()
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

3. **Use a simpler model**: Follow the instructions in the "Training with a Simpler Model" section

4. **Check for extreme values in your data**: Use the `check_data.py` and `check_ecg.py` scripts to inspect your data for outliers

### Missing or Invalid ECG Files

If you encounter errors related to ECG files:

1. **Check file format**: Ensure your ECG JSON files have the correct structure with a `leads` array containing 12 lead objects, each with a `signal` array

2. **Check file naming**: Ensure your ECG JSON files are named with the patient ID in the format `123_2022.json` (with underscore)

3. **Check patient ID matching**: Ensure the patient IDs in your clinical data Excel file match the ECG filenames (but with "/" instead of "_")

### Memory Issues

If you encounter memory issues during training:

1. **Reduce batch size**: Edit the configuration file to use a smaller batch size (e.g., 16 or 8 instead of 32)

2. **Reduce model complexity**: Use smaller output dimensions and hidden layer sizes

## Advanced Usage

### Custom Configuration

You can create custom configuration files to experiment with different model architectures and hyperparameters:

1. Create a new configuration file:
   ```bash
   cp config/model_config.json config/custom_config.json
   ```

2. Edit the configuration file to change:
   - Training parameters (batch_size, learning_rate, etc.)
   - Model architecture parameters (model_type, output_dim, etc.)

3. Train with the custom configuration:
   ```bash
   python src/train.py --config config/custom_config.json --device cuda
   ```

### Training on a Subset of Data

To train on a subset of the data (useful for debugging):

1. Create a subset of your ECG files:
   ```bash
   mkdir -p data/ecg_subset
   cp $(ls data/ecg/*.json | head -n 50) data/ecg_subset/
   ```

2. Create a custom configuration file that uses the subset:
   ```bash
   cp config/model_config.json config/subset_config.json
   ```

3. Edit `config/subset_config.json` to use the subset directory:
   ```json
   "ecg_data_dir": "data/ecg_subset"
   ```

4. Train with the subset configuration:
   ```bash
   python src/train.py --config config/subset_config.json --device cuda
   ```

### Using a Pre-trained Model

If you have a pre-trained model, you can use it for inference:

```bash
python src/evaluate.py --config config/model_config.json --model path/to/pretrained_model.pt --output results --device cuda
```

### Ensemble Prediction

You can create an ensemble prediction by averaging the predictions from multiple models:

1. Evaluate each model separately:
   ```bash
   for i in {1..5}; do
     python src/evaluate.py --config config/model_config.json --model models/model_fold_$i.pt --output results_fold_$i --device cuda
   done
   ```

2. Create a script to average the predictions:
   ```python
   import pandas as pd
   import numpy as np

   # Load predictions from each fold
   dfs = []
   for i in range(1, 6):
       df = pd.read_csv(f"results_fold_{i}/predictions.csv")
       dfs.append(df)

   # Create a DataFrame with patient IDs and true labels
   ensemble_df = dfs[0][['patient_id', 'true_label']].copy()

   # Average the predicted probabilities
   ensemble_df['predicted_probability'] = np.mean([df['predicted_probability'] for df in dfs], axis=0)

   # Create binary predictions
   ensemble_df['predicted_label'] = (ensemble_df['predicted_probability'] > 0.5).astype(int)

   # Save the ensemble predictions
   ensemble_df.to_csv("ensemble_predictions.csv", index=False)
   ```

3. Run the script:
   ```bash
   python ensemble_script.py
   ```