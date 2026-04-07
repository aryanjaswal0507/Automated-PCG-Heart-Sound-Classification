import argparse
import pandas as pd
import numpy as np
import os
from src.utils import load_pcg_data, load_labels, ensure_dirs
from src.features import extract_all_features_for_signal, get_feature_column_names
from src.models import evaluate_models, train_best_model
from src.optimization import SpiderMonkeyOptimizer, grid_search_tune

def run_pipeline(data_path, labels_path, output_dir, feature_file=None, n_samples=None):
    """Executes the PCG classification pipeline."""
    ensure_dirs([output_dir])
    
    # 1. Load Data/Features
    if feature_file and os.path.exists(feature_file):
        print(f"--- Loading Extracted Features from: {feature_file} ---")
        df_features = pd.read_csv(feature_file)
        # Select only numeric columns to avoid ID columns causing errors
        X_df = df_features.select_dtypes(include=[np.number])
        X = X_df.values
    elif os.path.exists(data_path):
        print(f"--- Loading Raw Data: {data_path} ---")
        signals = load_pcg_data(data_path)
        if n_samples: signals = signals[:n_samples]
        
        print("--- Extracting Features ---")
        processed_features = []
        for i, signal in enumerate(signals):
            if i % 100 == 0: print(f"Progress: {i}/{len(signals)}")
            processed_features.append(extract_all_features_for_signal(signal))
            
        df_features = pd.DataFrame(processed_features, columns=get_feature_column_names())
        # Filter numeric for extracted features as well (just in case)
        X = df_features.select_dtypes(include=[np.number]).values
        # Save to processed folder for next time
        df_features.to_csv(os.path.join(output_dir, "extracted_features.csv"), index=False)
    else:
        # Check if default features exist if data file is missing
        default_feat = "Dataset/ewt_stat_features_all.csv"
        if os.path.exists(default_feat):
            print(f"--- Info: Signal file not found at {data_path}. Using {default_feat} instead. ---")
            df_features = pd.read_csv(default_feat)
            X = df_features.select_dtypes(include=[np.number]).values
        else:
            raise FileNotFoundError(f"Neither signal file ({data_path}) nor feature file found.")

    # 2. Load Labels
    labels = load_labels(labels_path)
    if n_samples and len(labels) > n_samples:
        labels = labels[:n_samples]
    
    # Ensure X and y match in length
    X = X[:len(labels)]
    y = labels[:len(X)]
    
    print(f"Num Samples: {len(X)}")
    
    print("--- Evaluating Models (10-fold CV) ---")
    model_results = evaluate_models(X, y)
    for model_name, metrics in model_results.items():
        print(f"{model_name}: Accuracy = {metrics['accuracy_mean']:.4f} (+/- {metrics['accuracy_std']:.4f})")
        
    print("--- Optimizing Random Forest (SMO Meta-heuristic) ---")
    smo = SpiderMonkeyOptimizer(n_pop=10, n_iter=5)  # Reduced for demonstration
    best_params = smo.optimize(X, y)
    print(f"Best Params (SMO): {best_params}")
    
    print("--- Training Final Model ---")
    final_model = train_best_model(X, y, classifier_name='RandomForest', **best_params)
    
    print("--- Pipeline Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCG Heart Sound Classification Pipeline")
    parser.add_argument("--data", type=str, default="Dataset/hi.csv", help="Path to signals CSV")
    parser.add_argument("--labels", type=str, default="Dataset/labels.csv", help="Path to labels CSV")
    parser.add_argument("--features", type=str, default="Dataset/ewt_stat_features_all.csv", help="Path to pre-extracted features CSV")
    parser.add_argument("--output", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--test", action="store_true", help="Run on small subset for testing")
    
    args = parser.parse_args()
    
    n_samples = 20 if args.test else None
    run_pipeline(args.data, args.labels, args.output, feature_file=args.features, n_samples=n_samples)
