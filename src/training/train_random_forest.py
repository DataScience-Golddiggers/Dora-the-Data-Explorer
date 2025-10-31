"""
Random Forest Training Script

Basato su Test_08 - RandomForest.ipynb
Train Random Forest per classificazione binaria.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
import pickle
import json
import os
from datetime import datetime
from typing import Dict
import argparse


class RandomForestTrainer:
    """Trainer per modello Random Forest."""
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 15,
                 min_samples_split: int = 50,
                 min_samples_leaf: int = 20,
                 class_weight: str = 'balanced',
                 random_state: int = 42):
        """
        Inizializza trainer Random Forest.
        
        Args:
            n_estimators: Numero di alberi nella foresta
            max_depth: Profondità massima alberi
            min_samples_split: Minimo sample per split
            min_samples_leaf: Minimo sample per foglia
            class_weight: 'balanced' o None
            random_state: Seed per riproducibilità
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        
    def train(self,
              X_train: pd.DataFrame,
              y_train: pd.Series) -> RandomForestClassifier:
        """
        Addestra modello Random Forest.
        
        Args:
            X_train: Features training
            y_train: Target training
            
        Returns:
            Modello Random Forest addestrato
        """
        print("="*70)
        print("TRAINING RANDOM FOREST")
        print("="*70)
        
        if self.class_weight == 'balanced':
            print(f"\nClass weight: {self.class_weight} (bilanciamento automatico)")
        else:
            print(f"\nClass weight: None (dataset già bilanciato)")
        
        # Crea modello
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1
        )
        
        # Training
        self.model.fit(X_train, y_train)
        
        self.feature_names = X_train.columns.tolist()
        
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
        if self.model is None:
            raise ValueError("Modello non ancora trainato")
        
        # Predizioni
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        print("\n" + "="*70)
        print("PERFORMANCE TEST SET - Random Forest")
        print("="*70)
        
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=['Non-TP', 'TP'],
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
        
        print(f"\nPRECISION (TP):  {precision:.4f}")
        print(f"RECALL (TP):     {recall:.4f}")
        print(f"F1-SCORE (TP):   {f1:.4f}")
        print(f"ROC AUC:         {roc_auc:.4f} ⭐")
        
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
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Calcola feature importance.
        
        Returns:
            DataFrame con feature importance ordinato
        """
        if self.model is None:
            raise ValueError("Modello non ancora trainato")
        
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        return feature_importance
    
    def save(self, model_dir: str):
        """
        Salva modello e metriche.
        
        Args:
            model_dir: Directory dove salvare il modello
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Salva modello con pickle
        with open(f'{model_dir}/model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        # Salva feature importance
        feature_importance = self.get_feature_importance()
        feature_importance.to_csv(f'{model_dir}/feature_importance.csv', index=False)
        
        print(f"\n✅ Modello salvato in {model_dir}/")
        print("  - model.pkl")
        print("  - feature_importance.csv")
    
    @staticmethod
    def save_metrics(metrics: Dict, model_dir: str, model_name: str = "RandomForest_v2"):
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
    parser = argparse.ArgumentParser(description='Train Random Forest model')
    parser.add_argument('--data_dir', type=str, default='../data/processed_v3_balanced',
                       help='Directory con dati processati')
    parser.add_argument('--model_dir', type=str, default='../models/random_forest_v2',
                       help='Directory per salvare modello')
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=15)
    parser.add_argument('--min_samples_split', type=int, default=50)
    parser.add_argument('--min_samples_leaf', type=int, default=20)
    parser.add_argument('--no_class_weight', action='store_true',
                       help='Disabilita class_weight (per dataset bilanciato)')
    
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
    class_weight = None if args.no_class_weight else 'balanced'
    trainer = RandomForestTrainer(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        class_weight=class_weight
    )
    
    # Training
    trainer.train(X_train, y_train)
    
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
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'min_samples_split': args.min_samples_split,
        'min_samples_leaf': args.min_samples_leaf,
        'class_weight': str(class_weight)
    }
    
    # Feature importance
    feature_importance = trainer.get_feature_importance()
    metrics['top_10_features'] = feature_importance.head(10).to_dict('records')
    
    # Salva tutto
    trainer.save(args.model_dir)
    RandomForestTrainer.save_metrics(metrics, args.model_dir)
    
    print(f"\n{'='*70}")
    print(f"📊 ROC AUC Test: {metrics['test_roc_auc']:.4f}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
