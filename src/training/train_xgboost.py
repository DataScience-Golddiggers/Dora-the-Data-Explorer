"""
XGBoost Training Script

Basato su Test_03 - XGBoost_v2_Model.ipynb
Train XGBoost per classificazione binaria con class balancing.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
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
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pickle
import json
import os
from datetime import datetime
from typing import Dict, Tuple
import argparse


class XGBoostTrainer:
    """Trainer per modello XGBoost."""
    
    def __init__(self, 
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 n_estimators: int = 200,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 random_state: int = 42):
        """
        Inizializza trainer XGBoost.
        
        Args:
            max_depth: Profondità massima alberi
            learning_rate: Learning rate
            n_estimators: Numero di boosting rounds
            subsample: Subsample ratio per training
            colsample_bytree: Subsample ratio per colonne
            random_state: Seed per riproducibilità
        """
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        
    def train(self, 
              X_train: pd.DataFrame, 
              y_train: pd.Series,
              X_test: pd.DataFrame = None,
              y_test: pd.Series = None,
              use_scale_pos_weight: bool = True) -> xgb.XGBClassifier:
        """
        Addestra modello XGBoost.
        
        Args:
            X_train: Features training
            y_train: Target training
            X_test: Features test (opzionale, per early stopping)
            y_test: Target test (opzionale)
            use_scale_pos_weight: Se True, usa scale_pos_weight per bilanciamento
            
        Returns:
            Modello XGBoost addestrato
        """
        print("="*70)
        print("TRAINING XGBOOST")
        print("="*70)
        
        # Calcola scale_pos_weight per gestire imbalance
        if use_scale_pos_weight:
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            scale_pos_weight = neg_count / pos_count
            print(f"\nClass imbalance: {neg_count} negativi vs {pos_count} positivi")
            print(f"scale_pos_weight: {scale_pos_weight:.2f}")
        else:
            scale_pos_weight = 1.0
            print("\nScale_pos_weight disabilitato (dataset bilanciato)")
        
        # Crea modello
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            tree_method='hist',
            n_jobs=-1
        )
        
        # Training
        eval_set = [(X_test, y_test)] if X_test is not None and y_test is not None else None
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=True
        )
        
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
        print("PERFORMANCE TEST SET")
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
    
    def cross_validate(self, 
                       X_train: pd.DataFrame, 
                       y_train: pd.Series,
                       cv_folds: int = 5) -> Dict:
        """
        Esegue cross-validation.
        
        Args:
            X_train: Features training
            y_train: Target training
            cv_folds: Numero di fold
            
        Returns:
            Dizionario con risultati CV
        """
        if self.model is None:
            raise ValueError("Modello non ancora trainato")
        
        print(f"\nEsecuzione {cv_folds}-Fold Stratified Cross-Validation...")
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_scores = cross_val_score(
            self.model, X_train, y_train,
            cv=skf,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        print(f"\nCross-Validation ROC AUC Scores:")
        for i, score in enumerate(cv_scores, 1):
            print(f"  Fold {i}: {score:.4f}")
        
        print(f"\nMedia:  {cv_scores.mean():.4f}")
        print(f"Std:    {cv_scores.std():.4f}")
        
        return {
            'cv_roc_auc_mean': float(cv_scores.mean()),
            'cv_roc_auc_std': float(cv_scores.std()),
            'cv_scores': cv_scores.tolist()
        }
    
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
    
    def save(self, model_dir: str, dataset_info: Dict = None):
        """
        Salva modello e metriche.
        
        Args:
            model_dir: Directory dove salvare il modello
            dataset_info: Info sul dataset (opzionale)
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Salva modello XGBoost (formato nativo)
        self.model.save_model(f'{model_dir}/model.json')
        
        # Salva anche con pickle per compatibilità
        with open(f'{model_dir}/model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        # Salva feature importance
        feature_importance = self.get_feature_importance()
        feature_importance.to_csv(f'{model_dir}/feature_importance.csv', index=False)
        
        print(f"\n✅ Modello salvato in {model_dir}/")
        print("  - model.json")
        print("  - model.pkl")
        print("  - feature_importance.csv")
    
    @staticmethod
    def save_metrics(metrics: Dict, model_dir: str, model_name: str = "XGBoost_v2"):
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
    parser = argparse.ArgumentParser(description='Train XGBoost model')
    parser.add_argument('--data_dir', type=str, default='../data/processed_v3_balanced',
                       help='Directory con dati processati')
    parser.add_argument('--model_dir', type=str, default='../models/xgboost_v2',
                       help='Directory per salvare modello')
    parser.add_argument('--max_depth', type=int, default=6)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--n_estimators', type=int, default=200)
    parser.add_argument('--cv_folds', type=int, default=5)
    parser.add_argument('--no_scale_pos_weight', action='store_true',
                       help='Disabilita scale_pos_weight (per dataset bilanciato)')
    
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
    trainer = XGBoostTrainer(
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators
    )
    
    # Training
    trainer.train(X_train, y_train, X_test, y_test, 
                 use_scale_pos_weight=not args.no_scale_pos_weight)
    
    # Evaluation
    metrics = trainer.evaluate(X_test, y_test)
    
    # Cross-validation
    cv_metrics = trainer.cross_validate(X_train, y_train, cv_folds=args.cv_folds)
    metrics.update(cv_metrics)
    
    # Dataset info
    metrics['n_features'] = int(X_train.shape[1])
    metrics['n_train_samples'] = int(len(X_train))
    metrics['n_test_samples'] = int(len(X_test))
    metrics['class_distribution_train'] = {
        'class_0': int((y_train == 0).sum()),
        'class_1': int((y_train == 1).sum())
    }
    metrics['hyperparameters'] = {
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'n_estimators': args.n_estimators,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    
    # Feature importance
    feature_importance = trainer.get_feature_importance()
    metrics['top_10_features'] = feature_importance.head(10).to_dict('records')
    
    # Salva tutto
    trainer.save(args.model_dir)
    XGBoostTrainer.save_metrics(metrics, args.model_dir)
    
    print(f"\n{'='*70}")
    print(f"📊 ROC AUC Test: {metrics['test_roc_auc']:.4f}")
    print(f"📊 ROC AUC CV:   {metrics['cv_roc_auc_mean']:.4f} ± {metrics['cv_roc_auc_std']:.4f}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
