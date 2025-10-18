# TCC: Advanced Framework for Facial Emotion Recognition

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)

This repository contains a robust and modular framework for training, evaluating, and comparing deep learning models for Facial Emotion Recognition (FER). The architecture is designed for experimentation, allowing for easy integration of new datasets and models, and provides powerful features like cross-dataset evaluation and Test-Time Augmentation (TTA).

## Key Features

- **Multi-Dataset Support:** Currently supports FER2013, RAF-DB, and ExpW. The `data_loader.py` is structured to easily add more datasets.
- **Multiple Model Architectures:** Pre-configured for DenseNet121, ResNet50, and EfficientNet-B0 using `timm` and `torchvision`. Adding new models is trivial.
- **Automated Workflow:** A single command trains all specified models on all active datasets, saves the weights, and generates all evaluation assets.
- **Intelligent Training:**
    - **Pre-trained Model Caching:** Automatically skips training if a model's weights are already saved, loading them for evaluation instead.
    - **Early Stopping:** Prevents overfitting and saves time by monitoring validation accuracy.
    - **Data Balancing:** Uses Random Oversampling to combat class imbalance in training data.
- **Advanced Evaluation:**
    - **Test-Time Augmentation (TTA):** Improves prediction accuracy by averaging results over multiple augmented versions of test images.
    - **Cross-Dataset Evaluation:** Generates a heatmap to rigorously test model generalization by training on one dataset and evaluating on another.
- **Comprehensive Reporting:** Automatically generates and saves:
    - Detailed logs for each run.
    - Class distribution plots (before and after balancing).
    - Confusion matrices for each model/dataset pair.
    - A final summary table comparing the performance of all models. 

## Project Structure

The codebase is modularized for clarity and maintainability.

```

.
├── main.py             # Main execution script to run the entire pipeline
├── config.py           # Central configuration file for models, datasets, and hyperparameters
├── data_loader.py      # Handles loading, splitting, and balancing all datasets
├── model_utils.py      # Defines model creation functions and ensemble logic
├── training.py         # Contains the core training and evaluation loops
├── utils.py            # Utility functions for logging, plotting, and Early Stopping
|
├── fer2013/            # Directory for the FER2013 dataset
├── rafdb/              # Directory for the RAF-DB dataset
├── expw/               # Directory for the Expression in-the-Wild dataset
|
└── saidas/             # Output directory for logs, plots, and saved models
├── saved_models/
└── ...plots and logs

````

## Setup and Installation

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
````

### 2\. Install Dependencies

Create a file named `requirements.txt` with the following content:

```txt
torch
torchvision
pandas
scikit-learn
matplotlib
seaborn
Pillow
timm
imbalanced-learn
numpy
```

Then, install the packages:

```bash
pip install -r requirements.txt
```

### 3. Dataset Structure

This framework expects a specific directory structure for the datasets. Create the folders in the root of the project as shown below:

  - **FER2013:**
    ```
    ./fer2013/
    ├── train/
    |   ├── angry/
    |   ├── disgust/
    |   └── ...
    └── test/
        ├── angry/
        ├── disgust/
        └── ...
    ```
  - **RAF-DB:**
    ```
    ./rafdb/DATASET/
    ├── train/
    |   ├── 1/
    |   ├── 2/
    |   └── ...
    └── test/
        ├── 1/
        ├── 2/
        └── ...
    ```
  - **Expression in-the-Wild (ExpW):**
    ```
    ./expw/Expw-F/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── sad/
    ├── surprise/
    └── neutral/
    ```

## How to Use

### 1\. Configure Your Experiment

The primary control file is `config.py`. Open it to customize your run:

  - **Activate Datasets:** Add or remove datasets from the `ACTIVE_DATASETS` dictionary.
    ```python
    ACTIVE_DATASETS = {
        'RAF-DB': 'load_rafdb',
        'ExpW': 'load_expw',
        # 'FER2013': 'load_fer2013', # Deactivated by commenting out
    }
    ```
  - **Configure Models:** Add, remove, or modify models and their specific batch sizes in the `MODEL_CONFIG` dictionary to optimize GPU memory usage.
    ```python
    MODEL_CONFIG = {
        'densenet121': {'batch_size': 96},
        'resnet50': {'batch_size': 144},
        # 'efficientnet_b0': {'batch_size': 144},
    }
    ```
  - **Adjust Hyperparameters:** Change `EPOCHS`, `LEARNING_RATE`, and `PATIENCE` as needed.

### 2\. Run the Pipeline

Execute the main script from your terminal. The script will handle everything automatically.

```bash
python main.py
```

### 3\. Review the Outputs

All results are saved to the `saidas/` directory.

  - **Logs:** A timestamped log file (`execution_log_...txt`) will contain detailed information about each step.
  - **Saved Models:** Trained model weights (`.pth` files) are stored in `saidas/saved_models/`.
  - **Plots:** You will find PNG images for:
      - Class distributions for each dataset (`<dataset>_dist_original.png`, `<dataset>_dist_balanced.png`).
      - Confusion matrices for each trained model (`<dataset>_<model>_cm_TTA.png`).
      - A summary heatmap of the cross-dataset evaluation (`cross_dataset_evaluation.png`).
  - **Final Summary:** The console and log file will display a clean summary table of all model accuracies and training times at the end of the run.

## Example Outputs

The framework automatically generates key visualizations to analyze model performance.

**Confusion Matrix (TTA)**
*Shows detailed per-class performance for a given model on a test set.*

**Cross-Dataset Evaluation Heatmap**
*Provides critical insights into how well models generalize to unseen data from different domains.*

## Future Work

  - **Create a "Mega data set:** Create a dataset that combine all the others datasets, as a way of having a dataset that has all variants of data, like color in-the-wild images, gray lab-created images, etc.
  - **Add More Models:** Integrate newer architectures like ViT (Vision Transformers) or ConvNeXt variants.
  - **Advanced Augmentation:** Experiment with more sophisticated augmentation techniques like Mixup or CutMix.
  - **Hyperparameter Tuning:** Integrate a library like Optuna or Ray Tune to systematically find the best hyperparameters.
