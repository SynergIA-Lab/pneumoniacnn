"""
generate_supplementary_table.py
===============================

Script to aggregate the training results across all architectures
and generate a final CSV (and LaTeX format) that compares 
the Frozen (Stage 1) and Fine-tuned (Stage 2) performance.
"""

import os
import json
import pandas as pd
import numpy as np

def main():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    output_dir = config["output_dir"]
    architectures = config["architectures"]
    seeds = config["seeds"]
    
    table_data = []

    print("Aggregating results from all architectures and seeds...")
    
    for arch in architectures:
        arch_dir = os.path.join(output_dir, arch)
        if not os.path.exists(arch_dir):
            print(f"[WARNING] Results directory not found for {arch}. Skipping.")
            continue
            
        arch_frozen = []
        arch_finetuned = []

        # Recopilar datos de todas las semillas
        for seed in seeds:
            seed_csv = os.path.join(arch_dir, f"seed_{seed}", f"kfold_results_seed{seed}.csv")
            if not os.path.exists(seed_csv):
                continue
                
            df_seed = pd.read_csv(seed_csv)
            # Remove summary rows like 'mean' or 'std' if they exist in earlier runs
            if 'fold' in df_seed.columns:
                df_seed = df_seed[df_seed['fold'].apply(lambda x: str(x).isdigit())]
                
            frozen = df_seed[df_seed["protocol"] == "Frozen"]
            finetuned = df_seed[df_seed["protocol"] == "Fine-tuned"]
            
            arch_frozen.append(frozen)
            arch_finetuned.append(finetuned)

        if not arch_frozen or not arch_finetuned:
            continue
            
        # Unir todos los folds y seeds
        df_frozen_all = pd.concat(arch_frozen)
        df_finetuned_all = pd.concat(arch_finetuned)
        
        # Calcular media y desviación estándar
        metrics = ["f1", "auc", "mcc", "pr_auc"]
        
        frozen_mean = df_frozen_all[metrics].mean()
        frozen_std = df_frozen_all[metrics].std()
        
        ft_mean = df_finetuned_all[metrics].mean()
        ft_std = df_finetuned_all[metrics].std()
        
        # Formatear como "mean ± std"
        def fmt(mean, std):
            return f"{mean:.3f} ± {std:.3f}"
            
        table_data.append({
            "Architecture": arch,
            "Protocol": "Frozen",
            "F1": fmt(frozen_mean["f1"], frozen_std["f1"]),
            "AUC": fmt(frozen_mean["auc"], frozen_std["auc"]),
            "MCC": fmt(frozen_mean["mcc"], frozen_std["mcc"]),
            "PR-AUC": fmt(frozen_mean["pr_auc"], frozen_std["pr_auc"])
        })
        
        table_data.append({
            "Architecture": arch,
            "Protocol": "Fine-tuned",
            "F1": fmt(ft_mean["f1"], ft_std["f1"]),
            "AUC": fmt(ft_mean["auc"], ft_std["auc"]),
            "MCC": fmt(ft_mean["mcc"], ft_std["mcc"]),
            "PR-AUC": fmt(ft_mean["pr_auc"], ft_std["pr_auc"])
        })

    df_table = pd.DataFrame(table_data)
    
    out_csv = os.path.join(output_dir, "supplementary_table_finetuning.csv")
    df_table.to_csv(out_csv, index=False)
    
    print("\n" + "="*60)
    print("FINAL COMPARISON TABLE (FROZEN VS FINE-TUNED)")
    print("="*60)
    print(df_table.to_string(index=False))
    print(f"\nSaved to: {out_csv}")
    
    print("\nLaTeX Code Snippet:")
    print("-" * 40)
    for index, row in df_table.iterrows():
        prefix = f"\\multirow{{2}}{{*}}{{{row['Architecture']}}}" if row["Protocol"] == "Frozen" else ""
        print(f"{prefix} & {row['Protocol']} & {row['F1']} & {row['AUC']} & {row['MCC']} & {row['PR-AUC']} \\\\")
        if row["Protocol"] == "Fine-tuned":
            print("\\midrule")

if __name__ == "__main__":
    main()
