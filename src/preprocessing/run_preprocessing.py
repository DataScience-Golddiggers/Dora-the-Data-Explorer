"""
Main Preprocessing Script

Script principale per eseguire l'intera pipeline di preprocessing e data rebalancing.
Basato su Test_03 - FeatureEngineering_v3.ipynb e Test_11 - DataRebalancing.ipynb
"""

import pandas as pd
import os
import argparse
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.feature_engineering import FeatureEngineeringPipeline, prepare_for_modeling
from preprocessing.data_rebalancing import (
    add_minority_samples_from_test,
    undersample_majority_class,
    create_balanced_splits
)


def main():
    parser = argparse.ArgumentParser(description='Preprocessing completo per GUIDE dataset')
    parser.add_argument('--train_file', type=str, default='../data/GUIDE_Train.csv',
                       help='File CSV training raw')
    parser.add_argument('--test_file', type=str, default='../data/GUIDE_Test.csv',
                       help='File CSV test raw (per data augmentation)')
    parser.add_argument('--output_dir', type=str, default='../data/processed_v3_balanced',
                       help='Directory output per dati processati')
    parser.add_argument('--balance_strategy', type=str, default='augment',
                       choices=['none', 'augment', 'undersample', 'both'],
                       help='Strategia di bilanciamento: none, augment (add from test), undersample, both')
    parser.add_argument('--test_size', type=float, default=0.3,
                       help='Proporzione per test split')
    parser.add_argument('--save_pipeline', action='store_true',
                       help='Salva pipeline per inferenza')
    parser.add_argument('--pipeline_path', type=str, default='../models/preprocessing_pipeline.pkl',
                       help='Path per salvare pipeline')
    
    args = parser.parse_args()
    
    print("="*70)
    print("PREPROCESSING PIPELINE - GUIDE DATASET")
    print("="*70)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Training file: {args.train_file}")
    print(f"Balance strategy: {args.balance_strategy}")
    print(f"Output directory: {args.output_dir}")
    
    # ========================================
    # 1. FEATURE ENGINEERING SU TRAINING SET
    # ========================================
    print("\n" + "="*70)
    print("STEP 1: FEATURE ENGINEERING SU TRAINING SET")
    print("="*70)
    
    print(f"\nCaricamento {args.train_file}...")
    df_train = pd.read_csv(args.train_file)
    print(f"Dataset caricato: {df_train.shape[0]:,} righe")
    
    # Crea e fitta pipeline
    pipeline = FeatureEngineeringPipeline(
        alpha=2.0,
        beta=2.0,
        top_n_mitre=30,
        rare_verdict_threshold=100,
        rare_category_threshold=100
    )
    
    print("\nApplicazione feature engineering (fit_transform)...")
    incident_train = pipeline.fit_transform(df_train)
    
    print(f"\nDataset processato a livello Incident:")
    print(f"  Shape: {incident_train.shape}")
    print(f"  Features: {incident_train.shape[1] - 3}")  # -3 per ID e 2 target
    print(f"\nDistribuzione classi originale:")
    print(incident_train['BinaryIncidentGrade'].value_counts())
    print(incident_train['BinaryIncidentGrade'].value_counts(normalize=True).mul(100).round(2))
    
    # ========================================
    # 2. DATA REBALANCING (opzionale)
    # ========================================
    if args.balance_strategy != 'none':
        print("\n" + "="*70)
        print("STEP 2: DATA REBALANCING")
        print("="*70)
        
        if args.balance_strategy in ['augment', 'both']:
            print(f"\nCaricamento {args.test_file} per augmentation...")
            df_test_raw = pd.read_csv(args.test_file)
            print(f"Test set caricato: {df_test_raw.shape[0]:,} righe")
            
            print("\nAugmentation con campioni TruePositive dal test set...")
            incident_train = add_minority_samples_from_test(
                incident_train,
                df_test_raw,
                pipeline,
                n_samples=None,  # Bilanciamento completo
                random_state=42
            )
        
        if args.balance_strategy in ['undersample', 'both']:
            print("\nUndersampling classe maggioritaria...")
            incident_train = undersample_majority_class(
                incident_train,
                ratio=1.0,  # Bilanciamento perfetto
                random_state=42
            )
    
    # ========================================
    # 3. TRAIN/TEST SPLIT
    # ========================================
    print("\n" + "="*70)
    print("STEP 3: TRAIN/TEST SPLIT STRATIFICATO")
    print("="*70)
    
    X_train, X_test, y_train, y_test = create_balanced_splits(
        incident_train,
        test_size=args.test_size,
        random_state=42
    )
    
    # ========================================
    # 4. SALVATAGGIO
    # ========================================
    print("\n" + "="*70)
    print("STEP 4: SALVATAGGIO DATASET PROCESSATI")
    print("="*70)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    X_train.to_csv(f'{args.output_dir}/X_train.csv', index=False)
    X_test.to_csv(f'{args.output_dir}/X_test.csv', index=False)
    y_train.to_csv(f'{args.output_dir}/y_train.csv', index=False, header=['BinaryIncidentGrade'])
    y_test.to_csv(f'{args.output_dir}/y_test.csv', index=False, header=['BinaryIncidentGrade'])
    incident_train.to_csv(f'{args.output_dir}/incident_features.csv', index=False)
    
    print(f"\nDataset salvati in {args.output_dir}/")
    print(f"  - X_train.csv: {X_train.shape}")
    print(f"  - X_test.csv: {X_test.shape}")
    print(f"  - y_train.csv: {y_train.shape}")
    print(f"  - y_test.csv: {y_test.shape}")
    print(f"  - incident_features.csv: {incident_train.shape}")
    
    # Salva pipeline per inferenza
    if args.save_pipeline:
        print("\n" + "="*70)
        print("STEP 5: SALVATAGGIO PIPELINE PER INFERENZA")
        print("="*70)
        
        os.makedirs(os.path.dirname(args.pipeline_path), exist_ok=True)
        pipeline.save(args.pipeline_path)
        print(f"✅ Pipeline salvata: {args.pipeline_path}")
    
    # Riepilogo finale
    print("\n" + "="*70)
    print("RIEPILOGO PREPROCESSING")
    print("="*70)
    print(f"\nFeatures totali: {X_train.shape[1]}")
    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"\nDistribuzione finale training:")
    print(y_train.value_counts())
    print(y_train.value_counts(normalize=True).mul(100).round(2))
    print(f"\nImbalance ratio: {(y_train==0).sum()/(y_train==1).sum():.2f}:1")
    print("\n✅ Preprocessing completato con successo!")
    print("="*70)


if __name__ == '__main__':
    main()
