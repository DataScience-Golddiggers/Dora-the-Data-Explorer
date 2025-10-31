"""
Inference Pipeline

Modulo per applicare modelli trainati a nuovi dati.
Garantisce che le stesse trasformazioni del training vengano applicate.
"""

import pandas as pd
import numpy as np
import pickle
import torch
import json
import os
from typing import Dict, List, Union
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.feature_engineering import FeatureEngineeringPipeline, prepare_for_modeling


class ModelInference:
    """Classe per inferenza su nuovi dati."""
    
    def __init__(self, model_dir: str, model_type: str = 'xgboost'):
        """
        Inizializza inferenza.
        
        Args:
            model_dir: Directory contenente modello e pipeline
            model_type: Tipo di modello ('xgboost', 'random_forest', 'mlp')
        """
        self.model_dir = model_dir
        self.model_type = model_type.lower()
        
        self.model = None
        self.pipeline = None
        self.scaler = None  # Solo per MLP
        self.device = None  # Device per MLP
        
        self._load_model()
        self._load_pipeline()
    
    def _get_device_name(self) -> str:
        """Restituisce nome descrittivo del device (solo per MLP)."""
        if self.device is None:
            return 'N/A'
        if self.device.type == 'mps':
            return 'Apple Silicon GPU (MPS)'
        elif self.device.type == 'cuda':
            return 'NVIDIA GPU (CUDA)'
        else:
            return 'CPU'
    
    def _load_model(self):
        """Carica modello trainato."""
        if self.model_type == 'xgboost':
            import xgboost as xgb
            model_path = f'{self.model_dir}/model.pkl'
            if not os.path.exists(model_path):
                model_path = f'{self.model_dir}/model.json'
                if os.path.exists(model_path):
                    self.model = xgb.XGBClassifier()
                    self.model.load_model(model_path)
                else:
                    raise FileNotFoundError(f"Modello XGBoost non trovato in {self.model_dir}")
            else:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
        
        elif self.model_type == 'random_forest':
            model_path = f'{self.model_dir}/model.pkl'
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modello Random Forest non trovato: {model_path}")
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        
        elif self.model_type == 'mlp':
            model_path = f'{self.model_dir}/model.pth'
            scaler_path = f'{self.model_dir}/scaler.pkl'
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modello MLP non trovato: {model_path}")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler MLP non trovato: {scaler_path}")
            
            # Device selection: MPS (Apple Silicon) > CUDA > CPU
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
            
            self.model = torch.load(model_path, map_location=device)
            self.model.eval()
            self.device = device
            
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Log device info
            device_name = self._get_device_name()
            print(f"✅ Modello {self.model_type} caricato da {self.model_dir}")
            print(f"   Device: {device} ({device_name})")
        
        else:
            raise ValueError(f"Tipo modello non supportato: {self.model_type}")
            
        if self.model_type != 'mlp':
            print(f"✅ Modello {self.model_type} caricato da {self.model_dir}")
    
    def _load_pipeline(self):
        """Carica pipeline di preprocessing."""
        # Cerca pipeline nella directory del modello o in preprocessing/
        pipeline_paths = [
            f'{self.model_dir}/preprocessing_pipeline.pkl',
            '../models/preprocessing_pipeline.pkl',
            'models/preprocessing_pipeline.pkl'
        ]
        
        pipeline_path = None
        for path in pipeline_paths:
            if os.path.exists(path):
                pipeline_path = path
                break
        
        if pipeline_path is None:
            raise FileNotFoundError(
                "Pipeline di preprocessing non trovata. "
                "Assicurarsi di salvare la pipeline dopo il training con pipeline.save()"
            )
        
        self.pipeline = FeatureEngineeringPipeline.load(pipeline_path)
        print(f"✅ Pipeline caricata da {pipeline_path}")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predice classe per nuovi dati.
        
        Args:
            df: DataFrame raw (stesso formato di GUIDE_Train.csv)
            
        Returns:
            Array con predizioni (0 o 1)
        """
        # Applica preprocessing usando pipeline salvata
        incident_df = self.pipeline.transform(df)
        
        # Prepara features
        X, _ = prepare_for_modeling(incident_df)
        
        # Predizioni
        if self.model_type in ['xgboost', 'random_forest']:
            predictions = self.model.predict(X)
        
        elif self.model_type == 'mlp':
            # Standardizza con scaler salvato
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            
            with torch.no_grad():
                proba = self.model(X_tensor).cpu().numpy().flatten()
            predictions = (proba > 0.5).astype(int)
        
        return predictions
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predice probabilità per nuovi dati.
        
        Args:
            df: DataFrame raw (stesso formato di GUIDE_Train.csv)
            
        Returns:
            Array con probabilità per classe 1 (TruePositive)
        """
        # Applica preprocessing
        incident_df = self.pipeline.transform(df)
        
        # Prepara features
        X, _ = prepare_for_modeling(incident_df)
        
        # Probabilità
        if self.model_type in ['xgboost', 'random_forest']:
            proba = self.model.predict_proba(X)[:, 1]
        
        elif self.model_type == 'mlp':
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            
            with torch.no_grad():
                proba = self.model(X_tensor).cpu().numpy().flatten()
        
        return proba
    
    def predict_with_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predice con metadata (IncidentId, probabilità, etc).
        
        Args:
            df: DataFrame raw
            
        Returns:
            DataFrame con IncidentId, predizioni, probabilità
        """
        # Applica preprocessing
        incident_df = self.pipeline.transform(df)
        
        # Predizioni
        predictions = self.predict(df)
        probabilities = self.predict_proba(df)
        
        # Crea risultato
        result_df = pd.DataFrame({
            'IncidentId': incident_df['IncidentId'],
            'Predicted_Class': predictions,
            'Predicted_Proba': probabilities,
            'Predicted_Label': ['TruePositive' if p == 1 else 'Non-TruePositive' for p in predictions]
        })
        
        # Aggiungi ground truth se disponibile
        if 'BinaryIncidentGrade' in incident_df.columns:
            result_df['True_Class'] = incident_df['BinaryIncidentGrade'].values
            result_df['Correct'] = (result_df['Predicted_Class'] == result_df['True_Class'])
        
        return result_df


def compare_models(df: pd.DataFrame, model_configs: List[Dict]) -> pd.DataFrame:
    """
    Confronta predizioni di più modelli sugli stessi dati.
    
    Args:
        df: DataFrame raw
        model_configs: Lista di dict con 'model_dir' e 'model_type' e 'name'
        
    Returns:
        DataFrame con predizioni di tutti i modelli
    """
    results = None
    
    for config in model_configs:
        model_name = config.get('name', config['model_type'])
        print(f"\n{'='*70}")
        print(f"Caricamento modello: {model_name}")
        print(f"{'='*70}")
        
        inference = ModelInference(config['model_dir'], config['model_type'])
        
        # Predizioni
        predictions = inference.predict(df)
        probabilities = inference.predict_proba(df)
        
        # Prima iterazione: crea DataFrame base
        if results is None:
            incident_df = inference.pipeline.transform(df)
            results = pd.DataFrame({
                'IncidentId': incident_df['IncidentId']
            })
            
            if 'BinaryIncidentGrade' in incident_df.columns:
                results['True_Class'] = incident_df['BinaryIncidentGrade'].values
        
        # Aggiungi predizioni di questo modello
        results[f'{model_name}_Pred'] = predictions
        results[f'{model_name}_Proba'] = probabilities
    
    # Calcola agreement tra modelli se ci sono multiple predizioni
    pred_cols = [col for col in results.columns if col.endswith('_Pred')]
    if len(pred_cols) > 1:
        results['All_Agree'] = results[pred_cols].nunique(axis=1) == 1
        results['Majority_Vote'] = results[pred_cols].mode(axis=1)[0]
    
    return results


def batch_inference(input_file: str,
                    output_file: str,
                    model_dir: str,
                    model_type: str,
                    chunk_size: int = 10000):
    """
    Inferenza su file grande con processing a chunk.
    
    Args:
        input_file: Path al file CSV raw
        output_file: Path dove salvare predizioni
        model_dir: Directory modello
        model_type: Tipo modello
        chunk_size: Dimensione chunk per processing
    """
    print(f"Batch inference su {input_file}")
    print(f"Chunk size: {chunk_size}")
    
    # Carica modello
    inference = ModelInference(model_dir, model_type)
    
    # Process a chunk
    first_chunk = True
    
    for i, chunk_df in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
        print(f"\nProcessing chunk {i+1}...")
        
        # Predizioni
        result_chunk = inference.predict_with_metadata(chunk_df)
        
        # Salva (append dopo primo chunk)
        mode = 'w' if first_chunk else 'a'
        header = first_chunk
        result_chunk.to_csv(output_file, mode=mode, header=header, index=False)
        
        first_chunk = False
        print(f"Chunk {i+1} salvato ({len(result_chunk)} righe)")
    
    print(f"\n✅ Batch inference completata. Output: {output_file}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Inference su nuovi dati')
    parser.add_argument('--input_file', type=str, required=True,
                       help='File CSV con nuovi dati (formato GUIDE)')
    parser.add_argument('--output_file', type=str, required=True,
                       help='File CSV per salvare predizioni')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory contenente modello')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['xgboost', 'random_forest', 'mlp'],
                       help='Tipo di modello')
    parser.add_argument('--chunk_size', type=int, default=10000,
                       help='Dimensione chunk per batch processing')
    
    args = parser.parse_args()
    
    batch_inference(
        args.input_file,
        args.output_file,
        args.model_dir,
        args.model_type,
        args.chunk_size
    )
