"""
MLP (Neural Network) Training Script

Basato su Test_07 - NeuralNetwork_MLP.ipynb e Test_10 - MLP_Undersampling.ipynb
Train Multi-Layer Perceptron per classificazione binaria.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
import pickle
import json
import os
from datetime import datetime
from typing import Dict, Tuple
import argparse


class MLPClassifier(nn.Module):
    """Multi-Layer Perceptron per classificazione binaria."""
    
    def __init__(self, input_dim: int, hidden_dims: list = [128, 64, 32], dropout: float = 0.3):
        """
        Inizializza MLP.
        
        Args:
            input_dim: Numero di input features
            hidden_dims: Lista con dimensioni hidden layers
            dropout: Dropout rate
        """
        super(MLPClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class MLPTrainer:
    """Trainer per modello MLP."""
    
    def __init__(self,
                 hidden_dims: list = [128, 64, 32],
                 dropout: float = 0.3,
                 learning_rate: float = 0.001,
                 batch_size: int = 256,
                 epochs: int = 50,
                 early_stopping_patience: int = 10,
                 random_state: int = 42):
        """
        Inizializza trainer MLP.
        
        Args:
            hidden_dims: Lista con dimensioni hidden layers
            dropout: Dropout rate
            learning_rate: Learning rate
            batch_size: Batch size per training
            epochs: Numero di epoche
            early_stopping_patience: Patience per early stopping
            random_state: Seed per riproducibilità
        """
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        
        self.model = None
        self.scaler = None
        
        # Device selection: MPS (Apple Silicon) > CUDA > CPU
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.feature_names = None
        
        # Set seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
    
    def _get_device_name(self) -> str:
        """Restituisce nome descrittivo del device."""
        if self.device.type == 'mps':
            return 'Apple Silicon GPU (MPS)'
        elif self.device.type == 'cuda':
            return f'NVIDIA GPU (CUDA {torch.version.cuda})'
        else:
            return 'CPU'
    
    def prepare_data(self,
                     X_train: pd.DataFrame,
                     y_train: pd.Series,
                     X_test: pd.DataFrame = None,
                     y_test: pd.Series = None) -> Tuple:
        """
        Prepara dati per training: scaling e conversione a tensori.
        
        Args:
            X_train: Features training
            y_train: Target training
            X_test: Features test (opzionale)
            y_test: Target test (opzionale)
            
        Returns:
            Tuple con DataLoaders e tensori
        """
        # Standardizzazione (IMPORTANTE per MLP)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Converti a tensori
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)
        
        # Crea DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        test_loader = None
        X_test_tensor = None
        y_test_tensor = None
        
        if X_test is not None and y_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_tensor = torch.FloatTensor(X_test_scaled)
            y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1)
            
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, test_loader, X_test_tensor, y_test_tensor
    
    def train(self,
              X_train: pd.DataFrame,
              y_train: pd.Series,
              X_test: pd.DataFrame = None,
              y_test: pd.Series = None,
              use_class_weight: bool = True) -> nn.Module:
        """
        Addestra modello MLP.
        
        Args:
            X_train: Features training
            y_train: Target training
            X_test: Features test (per validation)
            y_test: Target test
            use_class_weight: Se True, usa pos_weight per bilanciamento
            
        Returns:
            Modello MLP addestrato
        """
        print("="*70)
        print("TRAINING MLP")
        print("="*70)
        
        self.feature_names = X_train.columns.tolist()
        
        # Prepara dati
        train_loader, test_loader, X_test_tensor, y_test_tensor = self.prepare_data(
            X_train, y_train, X_test, y_test
        )
        
        # Crea modello
        input_dim = X_train.shape[1]
        self.model = MLPClassifier(input_dim, self.hidden_dims, self.dropout).to(self.device)
        
        print(f"\nArchitettura MLP: {input_dim} → {' → '.join(map(str, self.hidden_dims))} → 1")
        print(f"Device: {self.device} ({self._get_device_name()})")
        print(f"Parametri: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Loss function con class weight
        if use_class_weight:
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            pos_weight = torch.FloatTensor([neg_count / pos_count]).to(self.device)
            print(f"\nClass weight: pos_weight={pos_weight.item():.2f}")
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            print("\nClass weight: None (dataset bilanciato)")
            criterion = nn.BCELoss()
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                
                if use_class_weight:
                    # Per BCEWithLogitsLoss, rimuovi sigmoid dall'output
                    outputs = torch.logit(outputs.clamp(1e-7, 1 - 1e-7))
                
                loss = criterion(outputs if use_class_weight else outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
            
            train_loss /= len(train_loader.dataset)
            
            # Validation phase
            if test_loader is not None:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for X_batch, y_batch in test_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        
                        outputs = self.model(X_batch)
                        
                        if use_class_weight:
                            outputs = torch.logit(outputs.clamp(1e-7, 1 - 1e-7))
                        
                        loss = criterion(outputs if use_class_weight else outputs, y_batch)
                        val_loss += loss.item() * X_batch.size(0)
                
                val_loss /= len(test_loader.dataset)
                
                print(f"Epoch [{epoch+1}/{self.epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Salva best model
                    best_model_state = self.model.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                        self.model.load_state_dict(best_model_state)
                        break
            else:
                print(f"Epoch [{epoch+1}/{self.epochs}] - Train Loss: {train_loss:.4f}")
        
        print("\n✅ Training completato!")
        return self.model
    
    def evaluate(self,
                 X_test: pd.DataFrame,
                 y_test: pd.Series) -> Dict:
        """
        Valuta performance del modello.
        
        Args:
            X_test: Features test
            y_test: Target test
            
        Returns:
            Dizionario con metriche
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Modello non ancora trainato")
        
        # Prepara dati
        X_test_scaled = self.scaler.transform(X_test)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
        
        # Predizioni
        self.model.eval()
        with torch.no_grad():
            y_pred_proba = self.model(X_test_tensor).cpu().numpy().flatten()
        
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        print("\n" + "="*70)
        print("PERFORMANCE TEST SET - MLP")
        print("="*70)
        
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=['Non-TP (0)', 'TruePositive (1)'],
            digits=4
        ))
        
        # Calcola metriche
        cm = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Metriche per classe 0
        precision_0 = precision_score(y_test, y_pred, pos_label=0)
        recall_0 = recall_score(y_test, y_pred, pos_label=0)
        f1_0 = f1_score(y_test, y_pred, pos_label=0)
        
        print(f"\nACCURACY:          {accuracy:.4f}")
        print(f"PRECISION (TP):    {precision:.4f}")
        print(f"RECALL (TP):       {recall:.4f}")
        print(f"F1-SCORE (TP):     {f1:.4f}")
        print(f"ROC AUC:           {roc_auc:.4f} ⭐")
        
        metrics = {
            'test_accuracy': float(accuracy),
            'test_precision': float(precision),
            'test_recall': float(recall),
            'test_f1_score': float(f1),
            'test_roc_auc': float(roc_auc),
            'per_class_metrics': {
                'class_0_non_tp': {
                    'precision': float(precision_0),
                    'recall': float(recall_0),
                    'f1_score': float(f1_0)
                },
                'class_1_tp': {
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1)
                }
            },
            'confusion_matrix': {
                'true_negatives': int(cm[0, 0]),
                'false_positives': int(cm[0, 1]),
                'false_negatives': int(cm[1, 0]),
                'true_positives': int(cm[1, 1])
            }
        }
        
        return metrics
    
    def save(self, model_dir: str):
        """
        Salva modello e scaler.
        
        Args:
            model_dir: Directory dove salvare il modello
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Salva model weights
        torch.save(self.model.state_dict(), f'{model_dir}/model_weights.pth')
        
        # Salva modello completo
        torch.save(self.model, f'{model_dir}/model.pth')
        
        # Salva scaler
        with open(f'{model_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"\n✅ Modello salvato in {model_dir}/")
        print("  - model_weights.pth")
        print("  - model.pth")
        print("  - scaler.pkl")
    
    @staticmethod
    def save_metrics(metrics: Dict, model_dir: str, model_name: str = "MLP"):
        """
        Salva metriche in JSON.
        
        Args:
            metrics: Dizionario con metriche
            model_dir: Directory dove salvare
            model_name: Nome del modello
        """
        metrics['model_name'] = model_name
        metrics['timestamp'] = datetime.now().isoformat()
        
        with open(f'{model_dir}/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"  - metrics.json")


def main():
    parser = argparse.ArgumentParser(description='Train MLP model')
    parser.add_argument('--data_dir', type=str, default='../data/processed_v3_balanced',
                       help='Directory con dati processati')
    parser.add_argument('--model_dir', type=str, default='../models/mlp_baseline',
                       help='Directory per salvare modello')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 64, 32])
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--no_class_weight', action='store_true',
                       help='Disabilita class weight (per dataset bilanciato)')
    
    args = parser.parse_args()
    
    # Carica dati
    print("Caricamento dati...")
    X_train = pd.read_csv(f'{args.data_dir}/X_train.csv')
    X_test = pd.read_csv(f'{args.data_dir}/X_test.csv')
    y_train = pd.read_csv(f'{args.data_dir}/y_train.csv')['BinaryIncidentGrade']
    y_test = pd.read_csv(f'{args.data_dir}/y_test.csv')['BinaryIncidentGrade']
    
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    
    # Crea trainer
    trainer = MLPTrainer(
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        early_stopping_patience=args.patience
    )
    
    # Training
    trainer.train(X_train, y_train, X_test, y_test,
                 use_class_weight=not args.no_class_weight)
    
    # Evaluation
    metrics = trainer.evaluate(X_test, y_test)
    
    # Dataset info
    metrics['n_features'] = int(X_train.shape[1])
    metrics['n_train_samples'] = int(len(X_train))
    metrics['n_test_samples'] = int(len(X_test))
    metrics['class_distribution_train'] = {
        'class_0': int((y_train == 0).sum()),
        'class_1': int((y_train == 1).sum())
    }
    metrics['hyperparameters'] = {
        'hidden_dims': args.hidden_dims,
        'dropout': args.dropout,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs
    }
    
    # Salva tutto
    trainer.save(args.model_dir)
    MLPTrainer.save_metrics(metrics, args.model_dir)
    
    print(f"\n{'='*70}")
    print(f"📊 ROC AUC Test: {metrics['test_roc_auc']:.4f}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
