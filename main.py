# main.py

import os
import time
import gc
import logging
import pandas as pd
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
import torch

# Import from our refactored modules
import config
import data_loader
import model_utils
import training
import utils
import numpy as np

def main():
    gc.collect()
    torch.cuda.empty_cache()
    # --- DYNAMIC LOGGING & DIRECTORY SETUP ---
    log_filename = f"execution_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    utils.setup_logging(config.OUTPUT_DIR, log_filename)
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)

    logging.info(f"Device: {config.DEVICE}")
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- 1. Load All Datasets ---
    datasets_data = {}
    for name, func_name in config.ACTIVE_DATASETS.items():
        logging.info(f"\n{'='*60}\nLoading Dataset: {name}\n{'='*60}")
        load_func = getattr(data_loader, func_name)
        (train_paths, train_labels), (test_paths, test_labels) = load_func()
        
        # CORRECTED CHECK: Use len() which works for both lists and numpy arrays.
        if len(train_paths) == 0:
            logging.warning(f"Skipping {name} due to loading failure or empty dataset.")
            continue
            
        datasets_data[name] = {
            'train': (train_paths, train_labels),
            'test': (test_paths, test_labels)
        }

    # Guard: if no datasets were loaded successfully, stop early
    if not datasets_data:
        logging.error("No datasets were loaded. Please verify dataset paths and ACTIVE_DATASETS in config.py.")
        return
        
    # --- Merge all training datasets into a single combined set ---
    logging.info(f"\n{'='*80}\nMerging all training datasets into a single unified dataset\n{'='*80}")
    
    # --- 2. Train models ---
    all_train_paths, all_train_labels = [], []
    results, trained_models = {}, {}
    num_classes = len(config.EMOTION_CLASSES)

    for name, data in datasets_data.items():
        train_paths, train_labels = data['train']
        all_train_paths.extend(train_paths)
        all_train_labels.extend(train_labels)

    logging.info(f"Total combined training samples: {len(all_train_paths)}")

    # balance the merged training data
    all_train_paths_b, all_train_labels_b = data_loader.balance_data_oversampling(
        all_train_paths, 
        np.array(all_train_labels)
    )

    utils.plot_class_distribution(
        all_train_labels_b, 
        config.EMOTION_CLASSES, 
        "Combined Dataset (Balanced)", 
        os.path.join(config.OUTPUT_DIR, "combined_train_distribution.png")
    )

    # Create one big training dataset
    merged_train_dataset = data_loader.LazyLoadDataset(
        all_train_paths_b,
        all_train_labels_b,
        transform=config.TRANSFORM_TRAIN_HEAVY
    )

    # Guard: if the merged dataset is empty, stop early to avoid training errors
    if len(merged_train_dataset) == 0:
        logging.error("Merged training dataset is empty after loading/balancing. Check dataset loaders and paths.")
        return

    for model_name, model_params in config.MODEL_CONFIG.items():
        torch.cuda.empty_cache()
        batch_size = model_params['batch_size']
        logging.info(f"\n--- Training unified model: {model_name} ---")

        model = model_utils.create_model(model_name, num_classes)
        model_path = os.path.join(config.MODEL_SAVE_DIR, f"merged_{model_name}.pth")

        if os.path.exists(model_path):
            logging.info(f"Found pre-trained merged model at {model_path}. Loading weights...")
            model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
            training_time = 0
            history = {}
        else:
            train_loader = DataLoader(merged_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
            
            # Create a small validation set from one of the datasets for early stopping
            val_paths, val_labels = datasets_data[list(datasets_data.keys())[0]]['test']
            val_dataset = data_loader.LazyLoadDataset(val_paths, val_labels, transform=config.TRANSFORM_TEST)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
            
            start_time = time.time()
            history = training.train_model(model, train_loader, val_loader)
            training_time = time.time() - start_time

            logging.info(f"Training complete. Saving merged model to {model_path}")
            torch.save(model.state_dict(), model_path)

        # Evaluate the merged model on each dataset’s test split
        model_results = {}
        for dataset_name, data in datasets_data.items():
            test_paths, test_labels = data['test']
            test_dataset = data_loader.LazyLoadDataset(test_paths, test_labels, transform=config.TRANSFORM_TEST)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

            acc, preds, true_lbls = training.evaluate_model(model, test_loader, use_tta=False)
            acc_tta, preds_tta, _ = training.evaluate_model(model, test_loader, use_tta=True)

            logging.info(f"{model_name} | Tested on {dataset_name} | Acc: {acc:.4f} | Acc (TTA): {acc_tta:.4f}")

            model_results[dataset_name] = {'acc': acc, 'acc_tta': acc_tta}
            utils.plot_confusion_matrix(
                true_lbls, preds_tta, config.EMOTION_CLASSES,
                f'CM {model_name} on {dataset_name} (Merged Model, TTA)',
                os.path.join(config.OUTPUT_DIR, f"merged_{model_name}_{dataset_name}_cm_TTA.png")
            )

        results[model_name] = model_results
        trained_models[model_name] = model
        
        del test_dataset, test_loader
        gc.collect()
        torch.cuda.empty_cache()

    # --- 3. Cross-Dataset Evaluation (Merged Models) ---
    active_datasets = list(datasets_data.keys())
    cross_results = {model_name: {test_ds: 0.0 for test_ds in active_datasets} for model_name in trained_models.keys()}

    logging.info("\n" + "="*80 + "\nCross-Dataset Evaluation (Merged Models)\n" + "="*80)
    for model_name, model in trained_models.items():
        logging.info(f"\nEvaluating merged model: {model_name} on all test datasets...")
        batch_size = config.MODEL_CONFIG[model_name]['batch_size']
        
        for test_ds in active_datasets:
            test_paths, test_labels = datasets_data[test_ds]['test']
            test_dataset = data_loader.LazyLoadDataset(test_paths, test_labels, transform=config.TRANSFORM_TEST)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
            
            acc, _, _ = training.evaluate_model(model, test_loader, use_tta=False)
            cross_results[model_name][test_ds] = acc
            logging.info(f"  {model_name} on {test_ds}: {acc:.4f}")
            
            del test_dataset, test_loader
        
        gc.collect()
        torch.cuda.empty_cache()

    utils.plot_cross_dataset_results(cross_results, os.path.join(config.OUTPUT_DIR, 'cross_dataset_evaluation.png'))
    
    # --- 4. Ensemble Evaluation (Merged Models) ---
    logging.info("\n" + "="*80 + "\nEnsemble Evaluation (All Merged Models Combined)\n" + "="*80)
    ensemble_results = {}
    
    # Create ensemble from all trained merged models
    all_models = list(trained_models.values())
    ensemble = model_utils.EnsembleModel(all_models, strategy='soft')
    
    for dataset_name, data in datasets_data.items():
        test_paths, test_labels = data['test']
        test_dataset = data_loader.LazyLoadDataset(test_paths, test_labels, transform=config.TRANSFORM_TEST)
        # Use batch size from first model (they should all be the same anyway)
        batch_size = config.MODEL_CONFIG[list(config.MODEL_CONFIG.keys())[0]]['batch_size']
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        logging.info(f"\nEvaluating ensemble on {dataset_name}...")
        acc, preds, true_lbls = ensemble.predict(test_loader)
        logging.info(f"Ensemble Accuracy on {dataset_name}: {acc:.4f}")
        
        ensemble_results[dataset_name] = {'accuracy': acc}
        
        # Plot confusion matrix for ensemble
        utils.plot_confusion_matrix(
            true_lbls, preds, config.EMOTION_CLASSES,
            f'Ensemble Confusion Matrix on {dataset_name}',
            os.path.join(config.OUTPUT_DIR, f"ensemble_{dataset_name}_cm.png")
        )
        
        # Classification report
        logging.info("\n" + classification_report(true_lbls, preds, target_names=config.EMOTION_CLASSES, digits=3))
        
        del test_dataset, test_loader
        gc.collect()
        torch.cuda.empty_cache()
    
    # Log ensemble summary
    logging.info("\n" + "="*80 + "\nEnsemble Results Summary\n" + "="*80)
    for dataset_name, metrics in ensemble_results.items():
        logging.info(f"{dataset_name}: {metrics['accuracy']:.4f}")
    
    # --- 5. Final Summary ---
    logging.info("\n" + "="*80 + "\nFinal Results Summary (Merged Models)\n" + "="*80)
    summary_data = []
    for model_name, test_results in results.items():
        for dataset_name, metrics in test_results.items():
            summary_data.append({
                'Model': model_name,
                'Tested on': dataset_name,
                'Accuracy': f"{metrics['acc']:.4f}",
                'Accuracy (TTA)': f"{metrics['acc_tta']:.4f}"
            })
    summary_df = pd.DataFrame(summary_data)
    logging.info("\n" + summary_df.to_string(index=False))

if __name__ == '__main__':
    main()