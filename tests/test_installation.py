"""
Test Suite per verificare l'installazione e il funzionamento della pipeline.

Esegui questo script per verificare che tutti i moduli siano correttamente installati
e funzionanti prima di iniziare il training.
"""

import sys
import os

# Aggiungi src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test: Verifica che tutti i moduli siano importabili."""
    print("="*70)
    print("TEST 1: IMPORT MODULI")
    print("="*70)
    
    try:
        print("✓ Testing pandas...")
        import pandas as pd
        print(f"  pandas version: {pd.__version__}")
        
        print("✓ Testing numpy...")
        import numpy as np
        print(f"  numpy version: {np.__version__}")
        
        print("✓ Testing scikit-learn...")
        import sklearn
        print(f"  scikit-learn version: {sklearn.__version__}")
        
        print("✓ Testing xgboost...")
        import xgboost as xgb
        print(f"  xgboost version: {xgb.__version__}")
        
        print("✓ Testing torch...")
        import torch
        print(f"  torch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        
        print("✓ Testing matplotlib...")
        import matplotlib
        print(f"  matplotlib version: {matplotlib.__version__}")
        
        print("✓ Testing seaborn...")
        import seaborn
        print(f"  seaborn version: {seaborn.__version__}")
        
        print("\n✅ Tutti i pacchetti base sono installati correttamente!")
        return True
        
    except ImportError as e:
        print(f"\n❌ ERRORE: {e}")
        print("Esegui: pip install -r requirements.txt")
        return False


def test_pipeline_modules():
    """Test: Verifica che i moduli della pipeline siano importabili."""
    print("\n" + "="*70)
    print("TEST 2: MODULI PIPELINE")
    print("="*70)
    
    try:
        print("✓ Testing preprocessing.feature_engineering...")
        from preprocessing.feature_engineering import FeatureEngineeringPipeline, prepare_for_modeling
        
        print("✓ Testing preprocessing.data_rebalancing...")
        from preprocessing.data_rebalancing import add_minority_samples_from_test, create_balanced_splits
        
        print("✓ Testing training.train_xgboost...")
        from training.train_xgboost import XGBoostTrainer
        
        print("✓ Testing training.train_random_forest...")
        from training.train_random_forest import RandomForestTrainer
        
        print("✓ Testing training.train_mlp...")
        from training.train_mlp import MLPTrainer
        
        print("✓ Testing inference.inference_pipeline...")
        from inference.inference_pipeline import ModelInference, compare_models
        
        print("\n✅ Tutti i moduli della pipeline sono importabili!")
        return True
        
    except ImportError as e:
        print(f"\n❌ ERRORE: {e}")
        print("Verifica la struttura della directory src/")
        return False


def test_data_availability():
    """Test: Verifica che i file dati siano presenti."""
    print("\n" + "="*70)
    print("TEST 3: FILE DATI")
    print("="*70)
    
    train_file = '../data/GUIDE_Train.csv'
    test_file = '../data/GUIDE_Test.csv'
    
    train_exists = os.path.exists(train_file)
    test_exists = os.path.exists(test_file)
    
    if train_exists:
        train_size = os.path.getsize(train_file) / (1024**3)  # GB
        print(f"✓ {train_file} presente ({train_size:.2f} GB)")
    else:
        print(f"❌ {train_file} non trovato")
    
    if test_exists:
        test_size = os.path.getsize(test_file) / (1024**3)  # GB
        print(f"✓ {test_file} presente ({test_size:.2f} GB)")
    else:
        print(f"❌ {test_file} non trovato")
    
    if train_exists and test_exists:
        print("\n✅ File dati presenti!")
        return True
    else:
        print("\n⚠️  Alcuni file dati sono mancanti.")
        print("Scarica i dataset GUIDE da: https://github.com/microsoft/msticpy/tree/master/docs/notebooks/data")
        return False


def test_feature_engineering_basic():
    """Test: Verifica funzionamento base del feature engineering."""
    print("\n" + "="*70)
    print("TEST 4: FEATURE ENGINEERING (BASIC)")
    print("="*70)
    
    try:
        import pandas as pd
        from preprocessing.feature_engineering import FeatureEngineeringPipeline
        
        # Crea dataset fittizio
        print("Creazione dataset di test fittizio...")
        df_test = pd.DataFrame({
            'IncidentId': [1, 1, 2, 2, 3, 3],
            'AlertId': [10, 10, 20, 20, 30, 30],
            'Id': [100, 101, 200, 201, 300, 301],
            'IncidentGrade': ['TruePositive', 'TruePositive', 'FalsePositive', 
                             'FalsePositive', 'BenignPositive', 'BenignPositive'],
            'AlertTitle': ['Phishing', 'Phishing', 'Malware', 'Malware', 'Spam', 'Spam'],
            'Timestamp': pd.date_range('2024-01-01', periods=6, freq='H'),
            'CountryCode': ['US', 'US', 'UK', 'UK', 'DE', 'DE'],
            'State': ['CA', 'CA', 'London', 'London', 'Berlin', 'Berlin'],
            'City': ['LA', 'LA', 'London', 'London', 'Berlin', 'Berlin'],
            'MitreTechniques': ['T1078', 'T1566', 'T1098', 'T1078', 'T1566', 'T1098'],
            'SuspicionLevel': ['High', 'High', 'Low', 'Low', 'Medium', 'Medium'],
            'EvidenceRole': ['Alert', 'Alert', 'Detection', 'Detection', 'Alert', 'Alert'],
            'ThreatFamily': ['Trojan', 'Trojan', 'Worm', 'Worm', 'Spam', 'Spam'],
            'AntispamDirection': ['Inbound', 'Inbound', 'Outbound', 'Outbound', 'Inbound', 'Inbound'],
            'ActionGranular': ['Block', 'Block', 'Alert', 'Alert', 'Quarantine', 'Quarantine'],
            'LastVerdict': ['Malicious', 'Malicious', 'Clean', 'Clean', 'Suspicious', 'Suspicious'],
            'ResourceType': ['Email', 'Email', 'File', 'File', 'Network', 'Network'],
            'Roles': ['User', 'User', 'Admin', 'Admin', 'User', 'User'],
            'ActionGrouped': ['Blocked', 'Blocked', 'Alerted', 'Alerted', 'Quarantined', 'Quarantined'],
            'EntityType': ['Account', 'Account', 'File', 'File', 'IP', 'IP'],
            'Category': ['InitialAccess', 'InitialAccess', 'Execution', 'Execution', 'Persistence', 'Persistence']
        })
        
        print("Inizializzazione pipeline...")
        pipeline = FeatureEngineeringPipeline(alpha=2.0, beta=2.0, top_n_mitre=3)
        
        print("Applicazione fit_transform...")
        incident_df = pipeline.fit_transform(df_test)
        
        print(f"\nDataset aggregato:")
        print(f"  Shape: {incident_df.shape}")
        print(f"  Incidents: {incident_df['IncidentId'].nunique()}")
        print(f"  Features: {incident_df.shape[1] - 3}")
        
        # Verifica features chiave
        required_features = ['NumAlerts', 'NumEvidences', 'SmoothedRisk_avg', 
                            'GeoLoc_freq_avg', 'Hour_First', 'Duration_seconds']
        missing_features = [f for f in required_features if f not in incident_df.columns]
        
        if missing_features:
            print(f"\n⚠️  Features mancanti: {missing_features}")
            return False
        
        print("\n✅ Feature engineering funziona correttamente!")
        return True
        
    except Exception as e:
        print(f"\n❌ ERRORE durante test feature engineering: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_trainers():
    """Test: Verifica inizializzazione dei trainer."""
    print("\n" + "="*70)
    print("TEST 5: INIZIALIZZAZIONE TRAINER")
    print("="*70)
    
    try:
        from training.train_xgboost import XGBoostTrainer
        from training.train_random_forest import RandomForestTrainer
        from training.train_mlp import MLPTrainer
        
        print("✓ Inizializzazione XGBoostTrainer...")
        xgb_trainer = XGBoostTrainer(max_depth=3, n_estimators=10)
        
        print("✓ Inizializzazione RandomForestTrainer...")
        rf_trainer = RandomForestTrainer(n_estimators=10)
        
        print("✓ Inizializzazione MLPTrainer...")
        mlp_trainer = MLPTrainer(hidden_dims=[32, 16], epochs=5)
        
        print("\n✅ Tutti i trainer si inizializzano correttamente!")
        return True
        
    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Esegue tutti i test."""
    print("\n" + "="*70)
    print("TEST SUITE - VERIFICA INSTALLAZIONE PIPELINE")
    print("="*70)
    print("\nQuesto script verifica che tutti i componenti siano installati correttamente.\n")
    
    results = {
        'imports': test_imports(),
        'pipeline_modules': test_pipeline_modules(),
        'data_availability': test_data_availability(),
        'feature_engineering': test_feature_engineering_basic(),
        'model_trainers': test_model_trainers()
    }
    
    print("\n" + "="*70)
    print("RIEPILOGO TEST")
    print("="*70)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<50} {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "="*70)
        print("🎉 TUTTI I TEST SUPERATI!")
        print("="*70)
        print("\nLa pipeline è pronta all'uso!")
        print("Consulta docs/PIPELINE_USAGE.md per iniziare.")
    else:
        print("\n" + "="*70)
        print("⚠️  ALCUNI TEST FALLITI")
        print("="*70)
        print("\nRisolvere i problemi prima di procedere.")
        print("Verifica requirements.txt e la struttura delle directory.")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
