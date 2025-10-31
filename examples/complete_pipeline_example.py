"""
Example: Uso Completo della Pipeline

Questo script mostra un esempio completo di utilizzo della pipeline:
1. Preprocessing e feature engineering
2. Training di tutti e tre i modelli
3. Inferenza e confronto
"""

import pandas as pd
import sys
import os

# Aggiungi src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing.feature_engineering import FeatureEngineeringPipeline, prepare_for_modeling
from preprocessing.data_rebalancing import add_minority_samples_from_test, create_balanced_splits
from training.train_xgboost import XGBoostTrainer
from training.train_random_forest import RandomForestTrainer
from training.train_mlp import MLPTrainer
from inference.inference_pipeline import ModelInference, compare_models


def main():
    print("="*70)
    print("ESEMPIO COMPLETO - PIPELINE GUIDE DATASET")
    print("="*70)
    
    # ========================================
    # 1. PREPROCESSING
    # ========================================
    print("\n" + "="*70)
    print("STEP 1: PREPROCESSING E FEATURE ENGINEERING")
    print("="*70)
    
    # Carica dati
    print("\nCaricamento dati...")
    df_train = pd.read_csv('../data/GUIDE_Train.csv')
    df_test_raw = pd.read_csv('../data/GUIDE_Test.csv')
    
    # Crea pipeline
    pipeline = FeatureEngineeringPipeline(
        alpha=2.0,
        beta=2.0,
        top_n_mitre=30
    )
    
    # Applica preprocessing al training
    print("Preprocessing training set...")
    incident_train = pipeline.fit_transform(df_train)
    
    # Data augmentation (opzionale)
    print("\nData augmentation da test set...")
    incident_train_balanced = add_minority_samples_from_test(
        incident_train,
        df_test_raw,
        pipeline,
        n_samples=10000,  # Aggiungi solo 10k campioni per esempio
        random_state=42
    )
    
    # Split train/test
    print("\nCreazione split stratificato...")
    X_train, X_test, y_train, y_test = create_balanced_splits(
        incident_train_balanced,
        test_size=0.3,
        random_state=42
    )
    
    # Salva pipeline
    os.makedirs('../models', exist_ok=True)
    pipeline.save('../models/preprocessing_pipeline.pkl')
    
    # ========================================
    # 2. TRAINING XGBOOST
    # ========================================
    print("\n" + "="*70)
    print("STEP 2: TRAINING XGBOOST")
    print("="*70)
    
    xgb_trainer = XGBoostTrainer(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100  # Ridotto per esempio
    )
    
    xgb_trainer.train(X_train, y_train, X_test, y_test, use_scale_pos_weight=False)
    xgb_metrics = xgb_trainer.evaluate(X_test, y_test)
    
    # Salva modello
    xgb_trainer.save('../models/xgboost_example')
    xgb_trainer.save_metrics(xgb_metrics, '../models/xgboost_example')
    
    # ========================================
    # 3. TRAINING RANDOM FOREST
    # ========================================
    print("\n" + "="*70)
    print("STEP 3: TRAINING RANDOM FOREST")
    print("="*70)
    
    rf_trainer = RandomForestTrainer(
        n_estimators=50,  # Ridotto per esempio
        max_depth=15,
        class_weight=None  # Dataset già bilanciato
    )
    
    rf_trainer.train(X_train, y_train)
    rf_metrics = rf_trainer.evaluate(X_test, y_test)
    
    # Salva modello
    rf_trainer.save('../models/random_forest_example')
    rf_trainer.save_metrics(rf_metrics, '../models/random_forest_example')
    
    # ========================================
    # 4. TRAINING MLP
    # ========================================
    print("\n" + "="*70)
    print("STEP 4: TRAINING MLP")
    print("="*70)
    
    mlp_trainer = MLPTrainer(
        hidden_dims=[64, 32],  # Rete più piccola per esempio
        epochs=10,  # Poche epoche per esempio
        batch_size=256
    )
    
    print(f"\nMLP utilizzerà: {mlp_trainer._get_device_name()}")
    mlp_trainer.train(X_train, y_train, X_test, y_test, use_class_weight=False)
    mlp_metrics = mlp_trainer.evaluate(X_test, y_test)
    
    # Salva modello
    mlp_trainer.save('../models/mlp_example')
    mlp_trainer.save_metrics(mlp_metrics, '../models/mlp_example')
    
    # ========================================
    # 5. CONFRONTO MODELLI
    # ========================================
    print("\n" + "="*70)
    print("STEP 5: CONFRONTO PERFORMANCE")
    print("="*70)
    
    results = pd.DataFrame({
        'Model': ['XGBoost', 'Random Forest', 'MLP'],
        'ROC AUC': [
            xgb_metrics['test_roc_auc'],
            rf_metrics['test_roc_auc'],
            mlp_metrics['test_roc_auc']
        ],
        'F1 Score': [
            xgb_metrics['test_f1_score'],
            rf_metrics['test_f1_score'],
            mlp_metrics['test_f1_score']
        ],
        'Precision': [
            xgb_metrics['test_precision'],
            rf_metrics['test_precision'],
            mlp_metrics['test_precision']
        ],
        'Recall': [
            xgb_metrics['test_recall'],
            rf_metrics['test_recall'],
            mlp_metrics['test_recall']
        ]
    })
    
    print("\n" + results.to_string(index=False))
    
    # Trova miglior modello
    best_model = results.loc[results['ROC AUC'].idxmax(), 'Model']
    best_auc = results['ROC AUC'].max()
    
    print(f"\n🏆 Miglior modello: {best_model} (ROC AUC: {best_auc:.4f})")
    
    # ========================================
    # 6. INFERENZA SU NUOVI DATI
    # ========================================
    print("\n" + "="*70)
    print("STEP 6: INFERENZA SU NUOVI DATI")
    print("="*70)
    
    # Simula nuovi dati (usiamo un subset del test set raw)
    df_new = df_test_raw.head(1000)
    
    print("\nInferenza con tutti i modelli...")
    
    model_configs = [
        {'model_dir': '../models/xgboost_example', 'model_type': 'xgboost', 'name': 'XGBoost'},
        {'model_dir': '../models/random_forest_example', 'model_type': 'random_forest', 'name': 'RandomForest'},
        {'model_dir': '../models/mlp_example', 'model_type': 'mlp', 'name': 'MLP'}
    ]
    
    comparison_results = compare_models(df_new, model_configs)
    
    print(f"\nRisultati inferenza (prime 10 righe):")
    print(comparison_results.head(10).to_string(index=False))
    
    # Calcola agreement
    if 'All_Agree' in comparison_results.columns:
        agreement_pct = comparison_results['All_Agree'].mean() * 100
        print(f"\nPercentuale agreement tra tutti i modelli: {agreement_pct:.2f}%")
    
    # Salva risultati
    comparison_results.to_csv('../predictions_comparison.csv', index=False)
    print("\n✅ Risultati salvati in predictions_comparison.csv")
    
    print("\n" + "="*70)
    print("ESEMPIO COMPLETATO CON SUCCESSO!")
    print("="*70)
    print("\nFile creati:")
    print("  - models/preprocessing_pipeline.pkl")
    print("  - models/xgboost_example/")
    print("  - models/random_forest_example/")
    print("  - models/mlp_example/")
    print("  - predictions_comparison.csv")
    print("\nConsulta docs/PIPELINE_USAGE.md per maggiori dettagli.")


if __name__ == '__main__':
    main()
