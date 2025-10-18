# utils.py

import os
import logging
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_acc):
        if self.best_score is None or val_acc > self.best_score + self.min_delta:
            self.best_score = val_acc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def setup_logging(log_dir, log_file):
    """Initializes logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized. Log file at: {log_path}")

def plot_class_distribution(labels, class_names, title, save_path):
    """Visualizes class distribution and saves the plot."""
    # CORRECTED CHECK: Explicitly check if the array is None or has a size of 0.
    if labels is None or labels.size == 0:
        logging.warning(f"No labels provided for plot '{title}'. Skipping.")
        return
        
    label_counts = Counter(labels)
    # Ensure keys are integers for proper indexing if they are not already
    sorted_labels = sorted([int(k) for k in label_counts.keys()])
    counts = [label_counts[key] for key in sorted_labels]
    # Handle potential IndexError if a label is out of bounds
    names = [class_names[key] if key < len(class_names) else f"Label {key}" for key in sorted_labels]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=names, y=counts)
    plt.title(title)
    plt.xlabel('Emotion')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.info(f"Class distribution plot saved to: {save_path}")

def plot_confusion_matrix(true_labels, predictions, class_names, title, save_path):
    """Plots a confusion matrix and saves it."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.info(f"Confusion matrix saved to: {save_path}")

def plot_cross_dataset_results(cross_results, save_path):
    """
    Plot cross-dataset results for merged models.
    cross_results structure: {model_name: {test_dataset: accuracy}}
    """
    models = list(cross_results.keys())
    datasets = list(next(iter(cross_results.values())).keys()) if cross_results else []
    
    if not models or not datasets:
        logging.warning("No cross-dataset results to plot.")
        return
    
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5.5), squeeze=False)
    
    for idx, model_name in enumerate(models):
        # Create data for this model: single row showing performance on each test dataset
        data = [[cross_results[model_name].get(test_ds, 0.0) for test_ds in datasets]]
        
        sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn', 
                    xticklabels=datasets, yticklabels=['Merged Model'], 
                    ax=axes[0, idx], vmin=0, vmax=1, cbar_kws={'label': 'Accuracy'})
        axes[0, idx].set_title(f'{model_name}\n(Merged Training → Test)')
        axes[0, idx].set_xlabel('Test Dataset')
        axes[0, idx].set_ylabel('')
        
    plt.tight_layout(pad=3.0)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.info(f"Cross-dataset evaluation plot saved to: {save_path}")