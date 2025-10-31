"""
Feature Engineering Pipeline v3 - GUIDE Dataset

Questo modulo implementa il preprocessing completo basato su:
- Test_03 - FeatureEngineering_v3.ipynb
- Test_11 - DataRebalancing.ipynb

Pipeline:
1. Caricamento e pulizia dati
2. Target binario (BinaryIncidentGrade)
3. SmoothedRisk per AlertTitle
4. GeoLoc_freq
5. Features temporali
6. Frequency encoding
7. One-hot encoding selettivo
8. MITRE top 30 techniques
9. Aggregazione a livello incident
"""

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import os
from typing import Dict, Tuple, Optional


class FeatureEngineeringPipeline:
    """Pipeline completa per feature engineering del dataset GUIDE."""
    
    def __init__(self, 
                 alpha: float = 2.0,
                 beta: float = 2.0,
                 top_n_mitre: int = 30,
                 rare_verdict_threshold: int = 100,
                 rare_category_threshold: int = 100):
        """
        Inizializza la pipeline.
        
        Args:
            alpha: Parametro alpha per Bayesian smoothing
            beta: Parametro beta per Bayesian smoothing
            top_n_mitre: Numero di top MITRE techniques da usare
            rare_verdict_threshold: Soglia per raggruppare verdetti rari
            rare_category_threshold: Soglia per raggruppare categorie rare
        """
        self.alpha = alpha
        self.beta = beta
        self.top_n_mitre = top_n_mitre
        self.rare_verdict_threshold = rare_verdict_threshold
        self.rare_category_threshold = rare_category_threshold
        
        # Statistiche salvate dal training set per applicare a test/inference
        self.alert_stats = None
        self.geo_freq = None
        self.freq_encoders = {}
        self.mlb = None
        self.top_mitre_techs = None
        self.rare_verdicts = None
        self.rare_categories = {}
        self.feature_names = None
        
    def create_binary_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea target binario: 1=TruePositive, 0=FalsePositive/BenignPositive.
        
        Args:
            df: DataFrame con colonna 'IncidentGrade'
            
        Returns:
            DataFrame con colonna 'BinaryIncidentGrade' aggiunta
        """
        df = df.copy()
        df['BinaryIncidentGrade'] = df['IncidentGrade'].apply(
            lambda x: 1 if x == 'TruePositive' else 0
        )
        return df
    
    def compute_smoothed_risk(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Calcola SmoothedRisk per AlertTitle usando Bayesian smoothing.
        
        Args:
            df: DataFrame con colonne 'AlertTitle' e 'BinaryIncidentGrade'
            fit: Se True, calcola statistiche da questo dataset (training).
                 Se False, applica statistiche salvate (test/inference).
                 
        Returns:
            DataFrame con colonna 'SmoothedRisk' aggiunta
        """
        df = df.copy()
        
        if fit:
            # Calcola statistiche dal training set
            alert_risk = df.groupby('AlertTitle')['BinaryIncidentGrade'].mean()
            alert_count = df.groupby('AlertTitle')['BinaryIncidentGrade'].count()
            
            self.alert_stats = pd.DataFrame({
                'Risk': alert_risk,
                'Count': alert_count
            })
            
            # Bayesian smoothing
            self.alert_stats['SmoothedRisk'] = (
                self.alert_stats['Risk'] * self.alert_stats['Count'] + self.alpha
            ) / (self.alert_stats['Count'] + self.alpha + self.beta)
            
        # Applica SmoothedRisk
        df = df.merge(
            self.alert_stats[['SmoothedRisk']], 
            left_on='AlertTitle', 
            right_index=True, 
            how='left'
        )
        
        # Fill NaN per AlertTitle non visti con media globale
        global_mean = self.alert_stats['SmoothedRisk'].mean()
        df['SmoothedRisk'] = df['SmoothedRisk'].fillna(global_mean)
        
        return df
    
    def compute_geoloc_freq(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Calcola frequenza normalizzata per location geografica.
        
        Args:
            df: DataFrame con colonne 'CountryCode', 'State', 'City'
            fit: Se True, calcola frequenze da questo dataset (training).
                 Se False, applica frequenze salvate (test/inference).
                 
        Returns:
            DataFrame con colonna 'GeoLoc_freq', senza colonne geografiche originali
        """
        df = df.copy()
        
        # Crea identificatore geografico
        df['GeoLoc'] = (
            df['CountryCode'].astype(str) + "_" + 
            df['State'].astype(str) + "_" + 
            df['City'].astype(str)
        )
        
        if fit:
            # Calcola frequenza normalizzata dal training set
            self.geo_freq = df['GeoLoc'].value_counts(normalize=True)
        
        # Applica frequenze
        df['GeoLoc_freq'] = df['GeoLoc'].map(self.geo_freq)
        
        # Fill NaN per location non viste con valore minimo
        df['GeoLoc_freq'] = df['GeoLoc_freq'].fillna(self.geo_freq.min())
        
        # Drop colonne geografiche
        df.drop(columns=['CountryCode', 'State', 'City', 'GeoLoc'], inplace=True)
        
        return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features temporali da colonna Timestamp.
        
        Args:
            df: DataFrame con colonna 'Timestamp'
            
        Returns:
            DataFrame con features temporali: month, hour, weekday, IsWeekend
        """
        df = df.copy()
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['month'] = df['Timestamp'].dt.month
        df['hour'] = df['Timestamp'].dt.hour
        df['weekday'] = df['Timestamp'].dt.weekday + 1
        df['IsWeekend'] = (df['Timestamp'].dt.dayofweek >= 5).astype(int)
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Gestisce missing values e raggruppa categorie rare.
        
        Args:
            df: DataFrame
            fit: Se True, identifica categorie rare dal training set
            
        Returns:
            DataFrame con missing values gestiti
        """
        df = df.copy()
        
        # Fill missing values
        df['Roles'] = df['Roles'].fillna('missing')
        df['ActionGrouped'] = df['ActionGrouped'].fillna('Missing')
        df['SuspicionLevel'] = df['SuspicionLevel'].fillna('Missing')
        df['LastVerdict'] = df['LastVerdict'].fillna('Missing')
        
        if fit:
            # Identifica verdetti rari
            verdict_counts = df['LastVerdict'].value_counts()
            self.rare_verdicts = verdict_counts[verdict_counts < self.rare_verdict_threshold].index
        
        # Raggruppa verdetti rari
        df['LastVerdict'] = df['LastVerdict'].replace(self.rare_verdicts, 'Other')
        
        return df
    
    def frequency_encode(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Applica frequency encoding a colonne ad alta cardinalità.
        
        Args:
            df: DataFrame
            fit: Se True, calcola frequenze dal training set
            
        Returns:
            DataFrame con frequency encoding applicato
        """
        df = df.copy()
        
        freq_encode_cols = [
            'ThreatFamily', 'AntispamDirection', 'ActionGranular',
            'LastVerdict', 'ResourceType', 'Roles', 'ActionGrouped', 
            'EntityType', 'Category'
        ]
        
        for col in freq_encode_cols:
            if col in df.columns:
                # Fill missing
                df[col] = df[col].fillna('Missing')
                
                if fit:
                    # Calcola frequenze dal training set
                    self.freq_encoders[col] = df[col].value_counts(normalize=True)
                
                # Applica frequency encoding
                df[f"{col}_freq"] = df[col].map(self.freq_encoders[col])
                
                # Fill NaN per categorie non viste con frequenza minima
                df[f"{col}_freq"] = df[f"{col}_freq"].fillna(self.freq_encoders[col].min())
                
                # Drop colonna originale
                df.drop(columns=col, inplace=True)
        
        return df
    
    def onehot_encode_selective(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Applica one-hot encoding selettivo a SuspicionLevel e EvidenceRole.
        
        Args:
            df: DataFrame
            fit: Se True, identifica categorie rare dal training set
            
        Returns:
            DataFrame con one-hot encoding applicato
        """
        df = df.copy()
        
        onehot_cols = ['SuspicionLevel', 'EvidenceRole']
        
        for col in onehot_cols:
            if col in df.columns:
                # Fill missing
                df[col] = df[col].fillna('Missing')
                
                if fit:
                    # Identifica categorie rare
                    counts = df[col].value_counts()
                    self.rare_categories[col] = counts[counts < self.rare_category_threshold].index
                
                # Raggruppa categorie rare
                df[col] = df[col].replace(self.rare_categories[col], 'Other')
                
                # One-hot encode (drop_first per evitare multicollinearità)
                df = pd.get_dummies(df, columns=[col], drop_first=True)
        
        return df
    
    def process_mitre_techniques(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Processa MITRE techniques: seleziona top N e applica one-hot encoding.
        
        Args:
            df: DataFrame con colonna 'MitreTechniques'
            fit: Se True, identifica top techniques dal training set
            
        Returns:
            DataFrame con MITRE features create
        """
        df = df.copy()
        
        # Split semicolon-separated string
        df['MitreList'] = df['MitreTechniques'].apply(
            lambda x: x.split(';') if pd.notna(x) else []
        )
        
        if fit:
            # Identifica top N tecniche dal training set
            all_techs = [tech for sublist in df['MitreList'] for tech in sublist]
            self.top_mitre_techs = [tech for tech, _ in Counter(all_techs).most_common(self.top_n_mitre)]
            
            # Crea MultiLabelBinarizer
            self.mlb = MultiLabelBinarizer(classes=self.top_mitre_techs)
            self.mlb.fit([self.top_mitre_techs])
        
        # Filtra liste per includere solo top techniques
        top_tech_set = set(self.top_mitre_techs)
        df['FilteredMitreList'] = df['MitreList'].apply(
            lambda x: [tech for tech in x if tech in top_tech_set]
        )
        
        # One-hot encode
        tech_matrix = pd.DataFrame(
            self.mlb.transform(df['FilteredMitreList']),
            columns=self.mlb.classes_, 
            index=df.index
        )
        
        # Merge e drop colonne originali
        df = pd.concat([df, tech_matrix], axis=1)
        df.drop(columns=['MitreTechniques', 'MitreList', 'FilteredMitreList'], inplace=True)
        
        return df
    
    def aggregate_to_incident_level(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggrega features da Evidence level a Incident level.
        
        Args:
            df: DataFrame a livello Evidence
            
        Returns:
            DataFrame aggregato a livello Incident
        """
        def get_mode(x):
            mode = x.mode()
            return mode[0] if len(mode) > 0 else x.iloc[0] if len(x) > 0 else None
        
        # Prepara aggregazioni
        agg_dict = {
            'BinaryIncidentGrade': 'first',
            'IncidentGrade': 'first',
            'AlertId': 'nunique',
            'Id': 'count',
            'SmoothedRisk': 'mean',
            'GeoLoc_freq': 'mean',
            'hour': ['min', 'max', 'mean'],
            'month': get_mode,
            'weekday': get_mode,
            'IsWeekend': 'max',
            'Timestamp': ['min', 'max'],
        }
        
        # Aggiungi frequency-encoded columns (media)
        freq_cols = [col for col in df.columns if col.endswith('_freq') and col != 'GeoLoc_freq']
        for col in freq_cols:
            agg_dict[col] = 'mean'
        
        # Aggiungi one-hot encoded columns (somma)
        onehot_cols = [col for col in df.columns if col.startswith(('SuspicionLevel_', 'EvidenceRole_'))]
        for col in onehot_cols:
            agg_dict[col] = 'sum'
        
        # Aggiungi MITRE columns (somma)
        mitre_cols = [col for col in df.columns if col.startswith('T') and len(col) <= 6]
        for col in mitre_cols:
            agg_dict[col] = 'sum'
        
        # Esegui aggregazione
        incident_agg = df.groupby('IncidentId').agg(agg_dict).reset_index()
        
        # Flatten colonne multi-livello
        incident_agg.columns = [
            '_'.join(col).strip('_') if isinstance(col, tuple) else col 
            for col in incident_agg.columns.values
        ]
        
        # Calcola durata
        incident_agg['Duration_seconds'] = (
            pd.to_datetime(incident_agg['Timestamp_max']) - 
            pd.to_datetime(incident_agg['Timestamp_min'])
        ).dt.total_seconds()
        
        # Rinomina colonne
        rename_map = {
            'AlertId_nunique': 'NumAlerts',
            'Id_count': 'NumEvidences',
            'SmoothedRisk_mean': 'SmoothedRisk_avg',
            'GeoLoc_freq_mean': 'GeoLoc_freq_avg',
            'hour_min': 'Hour_First',
            'hour_max': 'Hour_Last',
            'hour_mean': 'Hour_Avg',
            'BinaryIncidentGrade_first': 'BinaryIncidentGrade',
            'IncidentGrade_first': 'IncidentGrade',
        }
        
        incident_agg = incident_agg.rename(columns=rename_map)
        incident_agg = incident_agg.drop(columns=['Timestamp_min', 'Timestamp_max'], errors='ignore')
        
        return incident_agg
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applica pipeline completa e salva statistiche per inferenza (TRAINING).
        
        Args:
            df: DataFrame raw GUIDE
            
        Returns:
            DataFrame processato a livello Incident
        """
        # Pulizia
        df = df[df['IncidentGrade'].notna()].copy()
        
        # Target binario
        df = self.create_binary_target(df)
        
        # SmoothedRisk (fit=True calcola statistiche)
        df = self.compute_smoothed_risk(df, fit=True)
        
        # GeoLoc_freq (fit=True calcola frequenze)
        df = self.compute_geoloc_freq(df, fit=True)
        
        # Features temporali
        df = self.create_temporal_features(df)
        
        # Missing values (fit=True identifica categorie rare)
        df = self.handle_missing_values(df, fit=True)
        
        # Frequency encoding (fit=True calcola frequenze)
        df = self.frequency_encode(df, fit=True)
        
        # One-hot encoding (fit=True identifica categorie rare)
        df = self.onehot_encode_selective(df, fit=True)
        
        # MITRE techniques (fit=True identifica top N)
        df = self.process_mitre_techniques(df, fit=True)
        
        # Aggregazione a livello Incident
        incident_df = self.aggregate_to_incident_level(df)
        
        # Salva nomi features finali
        self.feature_names = [col for col in incident_df.columns 
                             if col not in ['IncidentId', 'BinaryIncidentGrade', 'IncidentGrade']]
        
        return incident_df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applica pipeline usando statistiche salvate dal training (TEST/INFERENCE).
        
        Args:
            df: DataFrame raw GUIDE
            
        Returns:
            DataFrame processato a livello Incident
        """
        if self.alert_stats is None:
            raise ValueError("Pipeline non ancora fittata. Esegui fit_transform() prima.")
        
        # Pulizia
        df = df[df['IncidentGrade'].notna()].copy()
        
        # Target binario
        df = self.create_binary_target(df)
        
        # SmoothedRisk (fit=False usa statistiche salvate)
        df = self.compute_smoothed_risk(df, fit=False)
        
        # GeoLoc_freq (fit=False usa frequenze salvate)
        df = self.compute_geoloc_freq(df, fit=False)
        
        # Features temporali
        df = self.create_temporal_features(df)
        
        # Missing values (fit=False usa categorie rare salvate)
        df = self.handle_missing_values(df, fit=False)
        
        # Frequency encoding (fit=False usa frequenze salvate)
        df = self.frequency_encode(df, fit=False)
        
        # One-hot encoding (fit=False usa categorie rare salvate)
        df = self.onehot_encode_selective(df, fit=False)
        
        # MITRE techniques (fit=False usa top N salvate)
        df = self.process_mitre_techniques(df, fit=False)
        
        # Aggregazione a livello Incident
        incident_df = self.aggregate_to_incident_level(df)
        
        # Allinea colonne con training set
        for col in self.feature_names:
            if col not in incident_df.columns:
                incident_df[col] = 0
        
        return incident_df
    
    def save(self, filepath: str):
        """
        Salva pipeline e statistiche per inferenza.
        
        Args:
            filepath: Percorso file .pkl
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Pipeline salvata: {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'FeatureEngineeringPipeline':
        """
        Carica pipeline salvata.
        
        Args:
            filepath: Percorso file .pkl
            
        Returns:
            Pipeline caricata
        """
        with open(filepath, 'rb') as f:
            pipeline = pickle.load(f)
        print(f"Pipeline caricata: {filepath}")
        return pipeline


def prepare_for_modeling(incident_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separa features (X) e target (y) per modeling.
    
    Args:
        incident_df: DataFrame a livello Incident
        
    Returns:
        Tuple (X, y) dove X sono le features e y il target
    """
    X = incident_df.drop(columns=['IncidentId', 'BinaryIncidentGrade', 'IncidentGrade'], errors='ignore')
    y = incident_df['BinaryIncidentGrade']
    
    # Gestisci missing (se presenti)
    X = X.fillna(-999)
    
    return X, y
