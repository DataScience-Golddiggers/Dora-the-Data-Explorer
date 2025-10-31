"""
Test veloce per verificare il supporto MPS/CUDA/CPU nel training e inferenza MLP.
"""

import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_device_detection():
    """Test: Verifica quale device sarà usato."""
    print("="*70)
    print("TEST DEVICE DETECTION")
    print("="*70)
    
    # Check MPS
    if torch.backends.mps.is_available():
        print("✅ MPS (Apple Silicon GPU) disponibile")
        device = torch.device('mps')
    elif torch.cuda.is_available():
        print("✅ CUDA (NVIDIA GPU) disponibile")
        device = torch.device('cuda')
    else:
        print("⚠️  Nessun acceleratore GPU disponibile, uso CPU")
        device = torch.device('cpu')
    
    print(f"\nDevice selezionato: {device}")
    
    # Test tensor allocation
    try:
        print(f"\nTest allocazione tensor su {device}...")
        x = torch.randn(100, 10).to(device)
        print(f"✅ Tensor allocato con successo: shape={x.shape}, device={x.device}")
        
        # Test operazione
        y = x @ x.T
        print(f"✅ Operazione matriciale completata: shape={y.shape}")
        
        return True
    except Exception as e:
        print(f"❌ ERRORE: {e}")
        return False


def test_mlp_trainer():
    """Test: Inizializzazione MLPTrainer con device detection."""
    print("\n" + "="*70)
    print("TEST MLP TRAINER DEVICE")
    print("="*70)
    
    try:
        from training.train_mlp import MLPTrainer
        
        trainer = MLPTrainer(
            hidden_dims=[32, 16],
            epochs=1,
            batch_size=32
        )
        
        print(f"✅ MLPTrainer inizializzato")
        print(f"   Device: {trainer.device}")
        print(f"   Device name: {trainer._get_device_name()}")
        
        return True
    except Exception as e:
        print(f"❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mlp_quick_training():
    """Test: Training veloce su dati fittizi."""
    print("\n" + "="*70)
    print("TEST MLP QUICK TRAINING")
    print("="*70)
    
    try:
        import pandas as pd
        import numpy as np
        from training.train_mlp import MLPTrainer
        
        # Crea dati fittizi
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        
        X_train = pd.DataFrame(np.random.randn(n_samples, n_features))
        y_train = pd.Series(np.random.randint(0, 2, n_samples))
        
        X_test = pd.DataFrame(np.random.randn(200, n_features))
        y_test = pd.Series(np.random.randint(0, 2, 200))
        
        print(f"Dataset fittizio: {X_train.shape[0]} train, {X_test.shape[0]} test")
        
        # Crea e addestra modello
        trainer = MLPTrainer(
            hidden_dims=[32, 16],
            epochs=2,
            batch_size=64
        )
        
        print(f"\nDevice utilizzato: {trainer._get_device_name()}")
        print("\nInizio training (2 epoche)...")
        
        trainer.train(X_train, y_train, X_test, y_test, use_class_weight=False)
        
        print("\n✅ Training completato con successo!")
        
        # Test predizione
        print("\nTest predizione...")
        metrics = trainer.evaluate(X_test, y_test)
        print(f"✅ ROC AUC: {metrics['test_roc_auc']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Esegue tutti i test."""
    print("\n" + "="*70)
    print("TEST SUITE - SUPPORTO MPS/CUDA/CPU PER MLP")
    print("="*70)
    print(f"\nPyTorch version: {torch.__version__}\n")
    
    results = {
        'device_detection': test_device_detection(),
        'mlp_trainer_init': test_mlp_trainer(),
        'mlp_quick_training': test_mlp_quick_training()
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
        print("\nIl supporto MPS/CUDA/CPU funziona correttamente!")
    else:
        print("\n" + "="*70)
        print("⚠️  ALCUNI TEST FALLITI")
        print("="*70)
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
