"""
Funzioni di supporto per l'analisi del dataset GUIDE
Microsoft Cybersecurity Incidents

Questo modulo contiene funzioni utili per:
- Preprocessing del dataset
- Feature engineering
- Analisi delle MITRE Techniques
- Gestione missing values
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


def load_guide_dataset(file_path: str, sample_frac: float = None) -> pd.DataFrame:
    """
    Carica il dataset GUIDE con opzione di campionamento

    Args:
        file_path: Percorso al file CSV
        sample_frac: Frazione del dataset da caricare (None = tutto)

    Returns:
        DataFrame con il dataset caricato
    """
    print(f"Caricamento dataset GUIDE da: {file_path}")

    if sample_frac:
        print(f"Campionamento: {sample_frac*100}% del dataset")
        # Conta le righe totali
        total_rows = sum(1 for _ in open(file_path)) - 1  # -1 per l'header
        skip_rows = np.random.choice(range(1, total_rows),
                                     size=int(total_rows * (1 - sample_frac)),
                                     replace=False)
        df = pd.read_csv(file_path, skiprows=skip_rows)
    else:
        df = pd.read_csv(file_path)

    print(f"✓ Dataset caricato: {df.shape[0]:,} righe, {df.shape[1]} colonne")
    return df


def clean_incident_grade(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rimuove record con IncidentGrade mancante (variabile target)

    Args:
        df: DataFrame con il dataset

    Returns:
        DataFrame pulito
    """
    initial_rows = len(df)
    df_clean = df[df['IncidentGrade'].notna()].copy()
    removed = initial_rows - len(df_clean)

    print(f"Rimozione record con IncidentGrade mancante:")
    print(f"  Prima: {initial_rows:,}")
    print(f"  Dopo: {len(df_clean):,}")
    print(f"  Rimossi: {removed:,} ({removed/initial_rows*100:.2f}%)")

    return df_clean


def extract_temporal_features(df: pd.DataFrame, timestamp_col: str = 'Timestamp') -> pd.DataFrame:
    """
    Estrae feature temporali dal timestamp

    Args:
        df: DataFrame con il dataset
        timestamp_col: Nome della colonna timestamp

    Returns:
        DataFrame con nuove colonne temporali
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Features temporali di base
    df['Hour'] = df[timestamp_col].dt.hour
    df['DayOfWeek'] = df[timestamp_col].dt.dayofweek
    df['DayOfMonth'] = df[timestamp_col].dt.day
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

    # Periodo del giorno
    df['TimeOfDay'] = pd.cut(df['Hour'],
                              bins=[0, 6, 12, 18, 24],
                              labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                              include_lowest=True)

    print(f"✓ Estratte {5} feature temporali")
    return df


def parse_mitre_techniques(df: pd.DataFrame,
                          mitre_col: str = 'MitreTechniques') -> pd.DataFrame:
    """
    Analizza e crea features dalle MITRE Techniques

    Args:
        df: DataFrame con il dataset
        mitre_col: Nome della colonna con le tecniche MITRE

    Returns:
        DataFrame con nuove colonne MITRE
    """
    df = df.copy()

    # Numero di tecniche per record
    df['MitreTechniques_Count'] = df[mitre_col].apply(
        lambda x: len(str(x).split(',')) if pd.notna(x) else 0
    )

    # Indicatore se ha tecniche MITRE
    df['HasMitreTechniques'] = (df[mitre_col].notna()).astype(int)

    # Estrai tutte le tecniche uniche per future one-hot encoding
    all_techniques = []
    for tech in df[mitre_col].dropna():
        if isinstance(tech, str):
            all_techniques.extend([t.strip() for t in tech.split(',')])

    unique_techniques = pd.Series(all_techniques).value_counts()
    print(f"✓ Trovate {len(unique_techniques)} tecniche MITRE uniche")
    print(f"✓ Top 5 tecniche: {list(unique_techniques.head().index)}")

    return df


def create_aggregated_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea feature aggregate a livello di Incident e Alert

    Args:
        df: DataFrame con il dataset

    Returns:
        DataFrame con feature aggregate
    """
    df = df.copy()

    # Aggregazioni a livello IncidentId
    if 'IncidentId' in df.columns:
        incident_agg = df.groupby('IncidentId').agg({
            'AlertId': 'nunique',
            'Id': 'count',
            'EntityType': 'nunique',
            'DetectorId': 'nunique',
            'Category': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
        }).rename(columns={
            'AlertId': 'NumAlerts_PerIncident',
            'Id': 'NumEvidences_PerIncident',
            'EntityType': 'NumEntityTypes_PerIncident',
            'DetectorId': 'NumDetectors_PerIncident',
            'Category': 'MostCommonCategory_PerIncident'
        })

        df = df.merge(incident_agg, on='IncidentId', how='left')
        print(f"✓ Aggiunte {len(incident_agg.columns)} feature aggregate per Incident")

    # Aggregazioni a livello AlertId
    if 'AlertId' in df.columns:
        alert_agg = df.groupby('AlertId').agg({
            'Id': 'count',
            'EntityType': 'nunique'
        }).rename(columns={
            'Id': 'NumEvidences_PerAlert',
            'EntityType': 'NumEntityTypes_PerAlert'
        })

        df = df.merge(alert_agg, on='AlertId', how='left')
        print(f"✓ Aggiunte {len(alert_agg.columns)} feature aggregate per Alert")

    return df


def create_missing_indicators(df: pd.DataFrame,
                              columns: List[str] = None) -> pd.DataFrame:
    """
    Crea indicatori binari per valori mancanti

    Args:
        df: DataFrame con il dataset
        columns: Lista di colonne da considerare (None = tutte con missing)

    Returns:
        DataFrame con indicatori missing
    """
    df = df.copy()

    if columns is None:
        # Identifica colonne con valori mancanti
        missing_cols = df.columns[df.isnull().any()].tolist()
    else:
        missing_cols = columns

    for col in missing_cols:
        if col in df.columns:
            df[f'{col}_IsMissing'] = df[col].isnull().astype(int)

    print(f"✓ Creati {len(missing_cols)} indicatori di missing values")
    return df


def get_incident_grade_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analizza la distribuzione di IncidentGrade

    Args:
        df: DataFrame con il dataset

    Returns:
        DataFrame con statistiche
    """
    grade_dist = df['IncidentGrade'].value_counts()
    grade_pct = df['IncidentGrade'].value_counts(normalize=True) * 100

    dist_df = pd.DataFrame({
        'Count': grade_dist,
        'Percentage': grade_pct
    })

    print("\n=== Distribuzione IncidentGrade ===")
    print(dist_df)
    print(f"\nTotale incidenti: {len(df):,}")

    return dist_df


def get_top_features_by_cardinality(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Identifica le colonne con maggiore cardinalità

    Args:
        df: DataFrame con il dataset
        n: Numero di top features da restituire

    Returns:
        DataFrame con statistiche di cardinalità
    """
    cardinality = []

    for col in df.columns:
        cardinality.append({
            'Column': col,
            'Unique_Values': df[col].nunique(),
            'Total_Values': df[col].count(),
            'Missing_Values': df[col].isnull().sum(),
            'Cardinality_Pct': (df[col].nunique() / df[col].count() * 100) if df[col].count() > 0 else 0
        })

    card_df = pd.DataFrame(cardinality).sort_values('Unique_Values', ascending=False).head(n)

    print(f"\n=== Top {n} Colonne per Cardinalità ===")
    print(card_df.to_string(index=False))

    return card_df


def prepare_for_modeling(df: pd.DataFrame,
                        target_col: str = 'IncidentGrade',
                        drop_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepara il dataset per il modeling

    Args:
        df: DataFrame con il dataset
        target_col: Nome della colonna target
        drop_cols: Lista di colonne da rimuovere

    Returns:
        Tuple (X, y) con features e target
    """
    df = df.copy()

    # Colonne da droppare di default
    default_drop = ['Id', 'Timestamp', 'IncidentId', 'AlertId']

    if drop_cols:
        cols_to_drop = list(set(default_drop + drop_cols))
    else:
        cols_to_drop = default_drop

    # Rimuovi colonne non necessarie
    cols_to_drop = [col for col in cols_to_drop if col in df.columns and col != target_col]

    # Separa features e target
    y = df[target_col].copy()
    X = df.drop(columns=cols_to_drop + [target_col])

    print(f"\n=== Preparazione per Modeling ===")
    print(f"Features (X): {X.shape}")
    print(f"Target (y): {y.shape}")
    print(f"Colonne rimosse: {cols_to_drop}")

    return X, y


def get_category_vs_grade_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analizza la relazione tra Category e IncidentGrade

    Args:
        df: DataFrame con il dataset

    Returns:
        DataFrame con statistiche incrociate
    """
    crosstab = pd.crosstab(df['Category'], df['IncidentGrade'], normalize='index') * 100
    crosstab = crosstab.round(2)
    crosstab['Total_Count'] = df.groupby('Category').size()
    crosstab = crosstab.sort_values('Total_Count', ascending=False)

    print("\n=== Category vs IncidentGrade (%) ===")
    print(crosstab.head(15))

    return crosstab


# Funzione di esempio per il workflow completo
def full_preprocessing_pipeline(file_path: str,
                               sample_frac: float = None) -> pd.DataFrame:
    """
    Pipeline completo di preprocessing

    Args:
        file_path: Percorso al file CSV
        sample_frac: Frazione del dataset (None = tutto)

    Returns:
        DataFrame preprocessato
    """
    print("=" * 80)
    print("INIZIO PREPROCESSING PIPELINE")
    print("=" * 80)

    # 1. Caricamento
    df = load_guide_dataset(file_path, sample_frac)

    # 2. Pulizia target
    df = clean_incident_grade(df)

    # 3. Feature temporali
    df = extract_temporal_features(df)

    # 4. MITRE Techniques
    df = parse_mitre_techniques(df)

    # 5. Aggregazioni
    df = create_aggregated_features(df)

    # 6. Missing indicators (solo per colonne importanti)
    important_cols = ['SuspicionLevel', 'LastVerdict', 'MitreTechniques']
    df = create_missing_indicators(df, important_cols)

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETATO")
    print("=" * 80)
    print(f"Dataset finale: {df.shape[0]:,} righe, {df.shape[1]} colonne")

    return df


if __name__ == "__main__":
    # Esempio di utilizzo
    print("Modulo di supporto per dataset GUIDE")
    print("Importa le funzioni necessarie nel tuo notebook:")
    print("\nfrom guide_utils import load_guide_dataset, extract_temporal_features, ...")

