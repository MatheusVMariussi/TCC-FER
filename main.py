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

def main():
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

    # --- 2. Train Models on Each Dataset ---
    results, trained_models = {}, {}
    num_classes = len(config.EMOTION_CLASSES)

    for dataset_name, data in datasets_data.items():
        logging.info(f"\n{'='*80}\nProcessing Dataset for Training: {dataset_name}\n{'='*80}")
        # ... (Balancing and plotting logic remains unchanged)
        (train_paths, train_labels) = data['train']
        (test_paths, test_labels) = data['test']
        utils.plot_class_distribution(train_labels, config.EMOTION_CLASSES, f"Original Dist - {dataset_name}", 
                                      os.path.join(config.OUTPUT_DIR, f"{dataset_name}_dist_original.png"))
        train_paths_b, train_labels_b = data_loader.balance_data_oversampling(train_paths, train_labels)
        utils.plot_class_distribution(train_labels_b, config.EMOTION_CLASSES, f"Balanced Dist - {dataset_name}",
                                      os.path.join(config.OUTPUT_DIR, f"{dataset_name}_dist_balanced.png"))
        train_dataset = data_loader.LazyLoadDataset(train_paths_b, train_labels_b, transform=config.TRANSFORM_TRAIN_HEAVY)
        test_dataset = data_loader.LazyLoadDataset(test_paths, test_labels, transform=config.TRANSFORM_TEST)
        
        results[dataset_name], trained_models[dataset_name] = {}, {}

        for model_name, model_params in config.MODEL_CONFIG.items():
            torch.cuda.empty_cache()
            batch_size = model_params['batch_size']
            logging.info(f"\n--- Processing {model_name} on {dataset_name} with Batch Size: {batch_size} ---")
            
            model = model_utils.create_model(model_name, num_classes)
            model_path = os.path.join(config.MODEL_SAVE_DIR, f"{dataset_name}_{model_name}.pth")

            # --- CHECK, LOAD, OR TRAIN-AND-SAVE LOGIC (NEW) ---
            if os.path.exists(model_path):
                logging.info(f"Found pre-trained model at {model_path}. Loading weights...")
                model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
                training_time = 0
                history = {} # No history available for pre-trained models
            else:
                logging.info(f"No pre-trained model found. Starting training...")
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
                
                start_time = time.time()
                history = training.train_model(model, train_loader, test_loader)
                training_time = time.time() - start_time
                
                logging.info(f"Training complete. Saving model to {model_path}")
                torch.save(model.state_dict(), model_path)
            # --- END OF NEW LOGIC ---

            logging.info(f"Evaluating {model_name}...")
            # The test_loader is now created inside the else block, let's define it for evaluation
            eval_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
            acc, preds, true_lbls = training.evaluate_model(model, eval_loader, use_tta=False)
            acc_tta, preds_tta, _ = training.evaluate_model(model, eval_loader, use_tta=True)
            logging.info(f"{model_name} | Accuracy: {acc:.4f} | Accuracy (TTA): {acc_tta:.4f}")

            results[dataset_name][model_name] = {
                'accuracy': acc, 'accuracy_tta': acc_tta,
                'training_time': training_time, 'history': history
            }
            trained_models[dataset_name][model_name] = {'model': model}

            logging.info("\n" + classification_report(true_lbls, preds_tta, target_names=config.EMOTION_CLASSES, digits=3))
            utils.plot_confusion_matrix(true_lbls, preds_tta, config.EMOTION_CLASSES, 
                                        f'CM {model_name} on {dataset_name} (TTA)',
                                        os.path.join(config.OUTPUT_DIR, f"{dataset_name}_{model_name}_cm_TTA.png"))
        
        del train_dataset, test_dataset, model
        gc.collect()
        torch.cuda.empty_cache()

    # --- 3. Cross-Dataset & Ensemble Evaluation ---
    active_datasets = list(datasets_data.keys())
    cross_results = {train_ds: {test_ds: {} for test_ds in active_datasets} for train_ds in active_datasets}

    logging.info("\n" + "="*80 + "\nCross-Dataset Evaluation\n" + "="*80)
    for train_ds in active_datasets:
        for test_ds in active_datasets:
            logging.info(f"\nEvaluating models from {train_ds} on {test_ds} data...")
            test_paths, test_labels = datasets_data[test_ds]['test']
            test_dataset = data_loader.LazyLoadDataset(test_paths, test_labels, transform=config.TRANSFORM_TEST)
            
            for model_name, model_info in trained_models[train_ds].items():
                batch_size = config.MODEL_CONFIG[model_name]['batch_size']
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
                acc, _, _ = training.evaluate_model(model_info['model'], test_loader)
                cross_results[train_ds][test_ds][model_name] = acc
                logging.info(f"  {model_name}: {acc:.4f}")

    utils.plot_cross_dataset_results(cross_results, os.path.join(config.OUTPUT_DIR, 'cross_dataset_evaluation.png'))
    
    # --- Final Summary ---
    logging.info("\n" + "="*80 + "\nFinal Results Summary\n" + "="*80)
    summary_data = []
    for ds, res_dict in results.items():
        for mdl, res in res_dict.items():
            summary_data.append({
                'Dataset': ds, 'Model': mdl,
                'Accuracy': res['accuracy'], 'Accuracy (TTA)': res['accuracy_tta'],
                'Training Time (s)': int(res['training_time'])
            })
    summary_df = pd.DataFrame(summary_data)
    logging.info("\n" + summary_df.to_string(index=False))

if __name__ == '__main__':
    main()