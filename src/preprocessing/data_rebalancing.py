"""
Data Rebalancing Module

Implementa strategie di bilanciamento del dataset basate su Test_11 - DataRebalancing.ipynb:
- Minority class augmentation da test set
- Undersampling
- SMOTE (opzionale)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional


def add_minority_samples_from_test(
    train_incident_df: pd.DataFrame,
    test_raw_df: pd.DataFrame,
    pipeline,
    n_samples: Optional[int] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Bilancia il training set aggiungendo campioni della classe minoritaria dal test set.
    
    Args:
        train_incident_df: DataFrame training a livello Incident
        test_raw_df: DataFrame raw del test set (GUIDE_Test.csv)
        pipeline: FeatureEngineeringPipeline già fittata sul training
        n_samples: Numero di campioni da aggiungere (None = bilanciamento completo)
        random_state: Seed per riproducibilità
        
    Returns:
        DataFrame bilanciato a livello Incident
    """
    # Calcola quanti campioni servono per bilanciamento
    class_counts = train_incident_df['BinaryIncidentGrade'].value_counts()
    majority_count = class_counts[0]
    minority_count = class_counts[1]
    samples_needed = majority_count - minority_count
    
    if n_samples is None:
        n_samples = samples_needed
    
    print(f"Classe maggioritaria (0): {majority_count:,}")
    print(f"Classe minoritaria (1): {minority_count:,}")
    print(f"Campioni da aggiungere: {n_samples:,}")
    
    # Applica preprocessing al test set usando statistiche del training
    test_incident_df = pipeline.transform(test_raw_df)
    
    # Estrai solo campioni TruePositive (classe 1)
    tp_samples = test_incident_df[test_incident_df['BinaryIncidentGrade'] == 1].copy()
    
    print(f"Campioni TruePositive disponibili nel test: {len(tp_samples):,}")
    
    # Sample casuale se abbiamo più campioni del necessario
    if len(tp_samples) > n_samples:
        tp_samples_selected = tp_samples.sample(n=n_samples, random_state=random_state)
        print(f"Campionamento casuale di {n_samples:,} campioni")
    else:
        tp_samples_selected = tp_samples
        print(f"Utilizzo di tutti i {len(tp_samples):,} campioni disponibili")
    
    # Allinea colonne
    train_cols = set(train_incident_df.columns)
    test_cols = set(tp_samples_selected.columns)
    
    missing_in_test = train_cols - test_cols
    missing_in_train = test_cols - train_cols
    
    if missing_in_test:
        for col in missing_in_test:
            tp_samples_selected[col] = 0
    
    if missing_in_train:
        for col in missing_in_train:
            train_incident_df[col] = 0
    
    # Allinea ordine colonne
    tp_samples_selected = tp_samples_selected[train_incident_df.columns]
    
    # Concatena
    balanced_df = pd.concat([train_incident_df, tp_samples_selected], axis=0, ignore_index=True)
    
    print(f"\nDataset bilanciato: {balanced_df.shape}")
    print(f"Nuova distribuzione:")
    print(balanced_df['BinaryIncidentGrade'].value_counts())
    
    return balanced_df


def undersample_majority_class(
    incident_df: pd.DataFrame,
    ratio: float = 1.0,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Undersampling della classe maggioritaria.
    
    Args:
        incident_df: DataFrame a livello Incident
        ratio: Rapporto desiderato majority/minority (1.0 = bilanciamento perfetto)
        random_state: Seed per riproducibilità
        
    Returns:
        DataFrame con undersampling applicato
    """
    df_majority = incident_df[incident_df['BinaryIncidentGrade'] == 0]
    df_minority = incident_df[incident_df['BinaryIncidentGrade'] == 1]
    
    # Calcola quanti campioni mantenere dalla classe maggioritaria
    n_minority = len(df_minority)
    n_majority_target = int(n_minority * ratio)
    
    # Sample dalla classe maggioritaria
    df_majority_sampled = df_majority.sample(n=n_majority_target, random_state=random_state)
    
    # Concatena
    balanced_df = pd.concat([df_majority_sampled, df_minority], axis=0, ignore_index=True)
    
    print(f"Undersampling completato:")
    print(f"  Classe 0: {len(df_majority):,} → {n_majority_target:,}")
    print(f"  Classe 1: {n_minority:,} (invariata)")
    print(f"  Totale: {len(balanced_df):,}")
    
    return balanced_df


def create_balanced_splits(
    incident_df: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Crea split train/test stratificato da dataset bilanciato.
    
    Args:
        incident_df: DataFrame bilanciato a livello Incident
        test_size: Proporzione per test set
        random_state: Seed per riproducibilità
        
    Returns:
        Tuple (X_train, X_test, y_train, y_test)
    """
    # Separa features e target
    X = incident_df.drop(columns=['IncidentId', 'BinaryIncidentGrade', 'IncidentGrade'], errors='ignore')
    y = incident_df['BinaryIncidentGrade']
    
    # Fill missing
    X = X.fillna(-999)
    
    # Split stratificato
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    print(f"\nSplit stratificato:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"\nDistribuzione y_train:")
    print(y_train.value_counts(normalize=True).mul(100).round(2))
    print(f"\nDistribuzione y_test:")
    print(y_test.value_counts(normalize=True).mul(100).round(2))
    
    return X_train, X_test, y_train, y_test
