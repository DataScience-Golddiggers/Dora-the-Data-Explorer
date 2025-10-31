# Pipeline Completa - GUIDE Dataset

Pipeline end-to-end per classificazione di incidenti di cybersecurity usando il dataset GUIDE.

## Struttura del Progetto

```
src/
├── preprocessing/          # Preprocessing e feature engineering
│   ├── feature_engineering.py
│   ├── data_rebalancing.py
│   └── run_preprocessing.py
├── training/              # Training dei modelli
│   ├── train_xgboost.py
│   ├── train_random_forest.py
│   └── train_mlp.py
└── inference/             # Inferenza su nuovi dati
    └── inference_pipeline.py
```

## Installazione

```bash
# Installa dipendenze
pip install -r requirements.txt
```

## Utilizzo

### 1. Preprocessing

Esegui la pipeline completa di preprocessing e feature engineering:

```bash
cd src/preprocessing

# Preprocessing base (senza bilanciamento)
python run_preprocessing.py \
    --train_file ../../data/GUIDE_Train.csv \
    --output_dir ../../data/processed_v3 \
    --balance_strategy none \
    --save_pipeline \
    --pipeline_path ../../models/preprocessing_pipeline.pkl

# Preprocessing con data augmentation dal test set
python run_preprocessing.py \
    --train_file ../../data/GUIDE_Train.csv \
    --test_file ../../data/GUIDE_Test.csv \
    --output_dir ../../data/processed_v3_balanced \
    --balance_strategy augment \
    --save_pipeline \
    --pipeline_path ../../models/preprocessing_pipeline.pkl

# Preprocessing con undersampling
python run_preprocessing.py \
    --train_file ../../data/GUIDE_Train.csv \
    --output_dir ../../data/processed_v3_undersampled \
    --balance_strategy undersample \
    --save_pipeline
```

**Output:**
- `data/processed_v3_balanced/X_train.csv` - Features training
- `data/processed_v3_balanced/X_test.csv` - Features test
- `data/processed_v3_balanced/y_train.csv` - Target training
- `data/processed_v3_balanced/y_test.csv` - Target test
- `data/processed_v3_balanced/incident_features.csv` - Dataset completo aggregato
- `models/preprocessing_pipeline.pkl` - Pipeline salvata per inferenza

### 2. Training Modelli

#### XGBoost

```bash
cd src/training

# Training con dataset bilanciato (no scale_pos_weight)
python train_xgboost.py \
    --data_dir ../../data/processed_v3_balanced \
    --model_dir ../../models/xgboost_v2 \
    --max_depth 6 \
    --learning_rate 0.1 \
    --n_estimators 200 \
    --cv_folds 5 \
    --no_scale_pos_weight

# Training con dataset non bilanciato (usa scale_pos_weight)
python train_xgboost.py \
    --data_dir ../../data/processed_v3 \
    --model_dir ../../models/xgboost_baseline \
    --max_depth 6 \
    --learning_rate 0.1 \
    --n_estimators 200 \
    --cv_folds 5
```

**Output:**
- `models/xgboost_v2/model.json` - Modello XGBoost
- `models/xgboost_v2/model.pkl` - Modello pickle
- `models/xgboost_v2/feature_importance.csv` - Feature importance
- `models/xgboost_v2/metrics.json` - Metriche di performance

#### Random Forest

```bash
# Training con class_weight='balanced'
python train_random_forest.py \
    --data_dir ../../data/processed_v3_balanced \
    --model_dir ../../models/random_forest_v2 \
    --n_estimators 100 \
    --max_depth 15 \
    --min_samples_split 50 \
    --min_samples_leaf 20

# Training senza class_weight (dataset bilanciato)
python train_random_forest.py \
    --data_dir ../../data/processed_v3_balanced \
    --model_dir ../../models/random_forest_balanced \
    --n_estimators 100 \
    --max_depth 15 \
    --no_class_weight
```

**Output:**
- `models/random_forest_v2/model.pkl`
- `models/random_forest_v2/feature_importance.csv`
- `models/random_forest_v2/metrics.json`

#### MLP (Neural Network)

```bash
# Training con class weight
python train_mlp.py \
    --data_dir ../../data/processed_v3_balanced \
    --model_dir ../../models/mlp_baseline \
    --hidden_dims 128 64 32 \
    --dropout 0.3 \
    --learning_rate 0.001 \
    --batch_size 256 \
    --epochs 50 \
    --patience 10

# Training senza class weight (dataset bilanciato)
python train_mlp.py \
    --data_dir ../../data/processed_v3_balanced \
    --model_dir ../../models/mlp_balanced \
    --hidden_dims 128 64 32 \
    --dropout 0.3 \
    --no_class_weight
```

**Output:**
- `models/mlp_baseline/model.pth` - Modello completo
- `models/mlp_baseline/model_weights.pth` - Solo weights
- `models/mlp_baseline/scaler.pkl` - StandardScaler per inferenza
- `models/mlp_baseline/metrics.json`

### 3. Inferenza

#### Inferenza Singola

```bash
cd src/inference

# Inferenza con XGBoost
python inference_pipeline.py \
    --input_file ../../data/GUIDE_Test.csv \
    --output_file ../../predictions_xgboost.csv \
    --model_dir ../../models/xgboost_v2 \
    --model_type xgboost

# Inferenza con Random Forest
python inference_pipeline.py \
    --input_file ../../data/GUIDE_Test.csv \
    --output_file ../../predictions_rf.csv \
    --model_dir ../../models/random_forest_v2 \
    --model_type random_forest

# Inferenza con MLP
python inference_pipeline.py \
    --input_file ../../data/GUIDE_Test.csv \
    --output_file ../../predictions_mlp.csv \
    --model_dir ../../models/mlp_baseline \
    --model_type mlp

# Batch inference per file grandi
python inference_pipeline.py \
    --input_file ../../data/GUIDE_Test.csv \
    --output_file ../../predictions.csv \
    --model_dir ../../models/xgboost_v2 \
    --model_type xgboost \
    --chunk_size 10000
```

#### Uso Programmatico

```python
from inference.inference_pipeline import ModelInference, compare_models
import pandas as pd

# Carica nuovo dataset
df_new = pd.read_csv('data/new_incidents.csv')

# Inferenza con singolo modello
inference = ModelInference(
    model_dir='models/xgboost_v2',
    model_type='xgboost'
)

# Predizioni
predictions = inference.predict(df_new)
probabilities = inference.predict_proba(df_new)

# Predizioni con metadata
results = inference.predict_with_metadata(df_new)
print(results.head())
# Output:
# IncidentId  Predicted_Class  Predicted_Proba  Predicted_Label
#      12345                1            0.892    TruePositive
#      12346                0            0.234    Non-TruePositive

# Confronta più modelli
model_configs = [
    {'model_dir': 'models/xgboost_v2', 'model_type': 'xgboost', 'name': 'XGBoost'},
    {'model_dir': 'models/random_forest_v2', 'model_type': 'random_forest', 'name': 'RandomForest'},
    {'model_dir': 'models/mlp_baseline', 'model_type': 'mlp', 'name': 'MLP'}
]

comparison = compare_models(df_new, model_configs)
print(comparison.head())
# Output include:
# IncidentId  True_Class  XGBoost_Pred  XGBoost_Proba  RandomForest_Pred  ...
```

## Features Create dalla Pipeline

### 1. Target Binario
- `BinaryIncidentGrade`: 1 = TruePositive, 0 = FalsePositive/BenignPositive

### 2. Risk Features
- `SmoothedRisk_avg`: Bayesian smoothing del risk per AlertTitle
- `NumWithSuspicion`: Numero di evidence con SuspicionLevel presente

### 3. Structural Features
- `NumAlerts`: Numero di alert unici per incident
- `NumEvidences`: Numero totale di evidence per incident
- `NumEntityTypes`: Diversità di entity types

### 4. Temporal Features
- `Hour_First`, `Hour_Last`, `Hour_Avg`: Statistiche orarie
- `month`, `weekday`: Features cicliche temporali
- `IsWeekend`: Flag weekend
- `Duration_seconds`: Durata incident in secondi

### 5. Geographic Features
- `GeoLoc_freq_avg`: Frequenza normalizzata della location geografica

### 6. Frequency Encoded Features
Frequency encoding per colonne ad alta cardinalità:
- `ThreatFamily_freq`
- `AntispamDirection_freq`
- `ActionGranular_freq`
- `LastVerdict_freq`
- `ResourceType_freq`
- `Roles_freq`
- `ActionGrouped_freq`
- `EntityType_freq`
- `Category_freq`

### 7. One-Hot Encoded Features
- `SuspicionLevel_*`: One-hot encoding di SuspicionLevel
- `EvidenceRole_*`: One-hot encoding di EvidenceRole

### 8. MITRE ATT&CK Techniques
Top 30 tecniche MITRE più frequenti (one-hot encoded):
- `T1078`, `T1566`, `T1098`, etc.

**Totale: ~43 features** (varia in base a one-hot encoding)

## Note Importanti

### Preprocessing
1. **SmoothedRisk**: Usa statistiche del training set per evitare data leakage
2. **GeoLoc_freq**: Location non viste nel training ricevono frequenza minima
3. **MITRE Techniques**: Solo top 30 tecniche dal training set sono usate
4. **Missing Values**: Gestiti con strategie specifiche per colonna

### Training
1. **XGBoost**: 
   - Usa `scale_pos_weight` per dataset non bilanciati
   - Disabilita con `--no_scale_pos_weight` per dataset bilanciati
2. **Random Forest**: 
   - Usa `class_weight='balanced'` per dataset non bilanciati
3. **MLP**: 
   - **RICHIEDE StandardScaler** (applicato automaticamente)
   - Salva scaler per inferenza
   - Usa `pos_weight` in BCEWithLogitsLoss per dataset non bilanciati
   - **Accelerazione GPU automatica**:
     - **Apple Silicon (M1/M2/M3)**: Usa MPS (Metal Performance Shaders)
     - **NVIDIA GPU**: Usa CUDA se disponibile
     - **Fallback CPU**: Se nessun acceleratore disponibile

### Inferenza
1. La pipeline salvata (`preprocessing_pipeline.pkl`) **DEVE** essere presente
2. Per MLP, il file `scaler.pkl` è necessario
3. Le trasformazioni applicate sono **identiche** a quelle del training
4. Allineamento automatico delle colonne per compatibilità
5. MLP usa automaticamente lo stesso acceleratore del training (MPS/CUDA/CPU)

## Metriche di Valutazione

Tutte le metriche sono salvate in `metrics.json`:
- `test_accuracy`: Accuracy sul test set
- `test_precision`: Precision per classe TruePositive
- `test_recall`: Recall per classe TruePositive
- `test_f1_score`: F1-Score per classe TruePositive
- `test_roc_auc`: ROC AUC (metrica principale) ⭐
- `per_class_metrics`: Metriche separate per entrambe le classi
- `confusion_matrix`: Matrice di confusione
- `cv_roc_auc_mean`: Media ROC AUC cross-validation (solo XGBoost)

## Troubleshooting

### Errore: "Pipeline non ancora fittata"
Assicurati di aver eseguito `run_preprocessing.py` con `--save_pipeline`

### Errore: "Scaler MLP non trovato"
Il file `scaler.pkl` è necessario per MLP. Riaddestra il modello.

### Colonne mancanti in inferenza
La pipeline allinea automaticamente le colonne. Verifica che il dataset raw abbia le stesse colonne del training.

### Out of Memory
Usa batch inference con `--chunk_size` più piccolo o riduci la dimensione del dataset.

## Workflow Completo

```bash
# 1. Preprocessing
cd src/preprocessing
python run_preprocessing.py \
    --train_file ../../data/GUIDE_Train.csv \
    --test_file ../../data/GUIDE_Test.csv \
    --output_dir ../../data/processed_v3_balanced \
    --balance_strategy augment \
    --save_pipeline \
    --pipeline_path ../../models/preprocessing_pipeline.pkl

# 2. Training XGBoost
cd ../training
python train_xgboost.py \
    --data_dir ../../data/processed_v3_balanced \
    --model_dir ../../models/xgboost_v2 \
    --no_scale_pos_weight

# 3. Training Random Forest
python train_random_forest.py \
    --data_dir ../../data/processed_v3_balanced \
    --model_dir ../../models/random_forest_v2 \
    --no_class_weight

# 4. Training MLP
python train_mlp.py \
    --data_dir ../../data/processed_v3_balanced \
    --model_dir ../../models/mlp_baseline \
    --no_class_weight

# 5. Inferenza
cd ../inference
python inference_pipeline.py \
    --input_file ../../data/GUIDE_Test.csv \
    --output_file ../../predictions_xgboost.csv \
    --model_dir ../../models/xgboost_v2 \
    --model_type xgboost
```

## References

- Dataset: GUIDE (Guided Response Dataset) - Microsoft Security AI Research
- MITRE ATT&CK Framework: https://attack.mitre.org/
- Metric principale: ROC AUC per classificazione binaria
