# Dora the Data Explorer - GUIDE Dataset Pipeline

Pipeline completa per classificazione di incidenti di cybersecurity usando il dataset GUIDE (Microsoft).

## 🚀 Quick Start

```bash
# 1. Installa dipendenze
pip install -r requirements.txt

# 2. Preprocessing
cd src/preprocessing
python run_preprocessing.py \
    --train_file ../../data/GUIDE_Train.csv \
    --test_file ../../data/GUIDE_Test.csv \
    --output_dir ../../data/processed_v3_balanced \
    --balance_strategy augment \
    --save_pipeline

# 3. Training (esempio con XGBoost)
cd ../training
python train_xgboost.py \
    --data_dir ../../data/processed_v3_balanced \
    --model_dir ../../models/xgboost_v2

# 4. Inferenza
cd ../inference
python inference_pipeline.py \
    --input_file ../../data/GUIDE_Test.csv \
    --output_file ../../predictions.csv \
    --model_dir ../../models/xgboost_v2 \
    --model_type xgboost
```

## 📁 Struttura del Progetto

```
Dora-the-Data-Explorer/
├── data/                           # Dataset
│   ├── GUIDE_Train.csv            # Training raw
│   ├── GUIDE_Test.csv             # Test raw
│   ├── processed_v3/              # Dati processati (no balancing)
│   └── processed_v3_balanced/     # Dati processati + bilanciati
├── src/                            # Codice sorgente
│   ├── preprocessing/             # Feature engineering e balancing
│   │   ├── feature_engineering.py
│   │   ├── data_rebalancing.py
│   │   └── run_preprocessing.py
│   ├── training/                  # Training modelli
│   │   ├── train_xgboost.py
│   │   ├── train_random_forest.py
│   │   └── train_mlp.py
│   └── inference/                 # Inferenza
│       └── inference_pipeline.py
├── models/                         # Modelli trainati
│   ├── preprocessing_pipeline.pkl # Pipeline per inferenza
│   ├── xgboost_v2/
│   ├── random_forest_v2/
│   └── mlp_baseline/
├── notebook/                       # Jupyter notebooks (sviluppo)
├── docs/                          # Documentazione
│   ├── PIPELINE_USAGE.md         # Documentazione dettagliata
│   └── QUICK_START.md
├── examples/                      # Esempi di utilizzo
│   └── complete_pipeline_example.py
└── requirements.txt               # Dipendenze Python
```

## 🔧 Componenti Principali

### 1. Preprocessing (`src/preprocessing/`)

**Feature Engineering Pipeline:**
- Target binario (TruePositive vs Non-TruePositive)
- SmoothedRisk con Bayesian smoothing
- Features temporali (hour, day, weekend, duration)
- GeoLoc_freq (frequenza location geografica)
- Frequency encoding per colonne ad alta cardinalità
- One-hot encoding selettivo (SuspicionLevel, EvidenceRole)
- MITRE ATT&CK top 30 techniques
- Aggregazione Evidence → Incident level

**Data Rebalancing:**
- Minority class augmentation dal test set
- Undersampling della classe maggioritaria
- Split stratificato train/test

### 2. Training (`src/training/`)

**Modelli implementati:**
- **XGBoost**: Gradient boosting con `scale_pos_weight` per class imbalance
- **Random Forest**: Ensemble di decision trees con `class_weight='balanced'`
- **MLP (Neural Network)**: Multi-layer perceptron con dropout e batch normalization

Tutti i modelli salvano:
- Modello trainato (`.pkl`, `.pth`, `.json`)
- Feature importance (XGBoost, Random Forest)
- Metriche complete (`metrics.json`)
- Scaler per normalizzazione (MLP)

### 3. Inference (`src/inference/`)

**Funzionalità:**
- Inferenza singola o batch
- Confronto tra più modelli
- Predizioni con metadata (probabilità, IncidentId)
- Applicazione automatica delle trasformazioni del training

## 📊 Features (43 totali)

### Structural
- `NumAlerts`: Numero alert per incident
- `NumEvidences`: Numero evidence per incident

### Temporal
- `Hour_First`, `Hour_Last`, `Hour_Avg`
- `month`, `weekday`, `IsWeekend`
- `Duration_seconds`

### Risk
- `SmoothedRisk_avg`: Bayesian smoothing per AlertTitle
- `GeoLoc_freq_avg`: Frequenza location

### Frequency Encoded (9 features)
- `ThreatFamily_freq`, `EntityType_freq`, `Category_freq`, etc.

### One-Hot Encoded
- `SuspicionLevel_*`: ~5 features
- `EvidenceRole_*`: ~3 features

### MITRE ATT&CK
- Top 30 tecniche più frequenti (e.g., `T1078`, `T1566`)

## 📈 Metriche di Valutazione

**Metrica principale:** ROC AUC (Area Under ROC Curve) ⭐

**Altre metriche:**
- Accuracy
- Precision, Recall, F1-Score (per entrambe le classi)
- Confusion Matrix
- Cross-validation scores (XGBoost)

## 🎯 Best Practices

### Preprocessing
✅ Usa `fit_transform()` solo sul training set  
✅ Usa `transform()` per test/inference  
✅ Salva sempre la pipeline per inferenza  
✅ SmoothedRisk usa statistiche del training (no data leakage)

### Training
✅ XGBoost: `scale_pos_weight` per dataset non bilanciati  
✅ Random Forest: `class_weight='balanced'` per imbalance  
✅ MLP: **Richiede StandardScaler** (applicato automaticamente)  
✅ Cross-validation con `StratifiedKFold`

### Inferenza
✅ Pipeline di preprocessing **DEVE** essere salvata  
✅ Per MLP, `scaler.pkl` è necessario  
✅ Le trasformazioni sono identiche al training  
✅ Usa batch inference per file grandi

## 📖 Documentazione

- **Documentazione completa**: [`docs/PIPELINE_USAGE.md`](docs/PIPELINE_USAGE.md)
- **Quick start**: [`docs/QUICK_START.md`](docs/QUICK_START.md)
- **Esempio completo**: [`examples/complete_pipeline_example.py`](examples/complete_pipeline_example.py)

## 🔬 Notebook di Sviluppo

I notebook originali da cui è stata estratta la pipeline:
- `Test_03 - FeatureEngineering_v3.ipynb`: Feature engineering
- `Test_11 - DataRebalancing.ipynb`: Data augmentation
- `Test_03 - XGBoost_v2_Model.ipynb`: XGBoost training
- `Test_08 - RandomForest.ipynb`: Random Forest training
- `Test_07 - NeuralNetwork_MLP.ipynb`: MLP training

## 🛠️ Requisiti

- Python 3.8+
- pandas, numpy, scikit-learn
- xgboost
- PyTorch (per MLP)
- matplotlib, seaborn (visualizzazioni)

Vedi [`requirements.txt`](requirements.txt) per la lista completa.

## 🤝 Contributi

Progetto sviluppato da **DataScience-Golddiggers** per il corso di Data Science, UnivPM.

## 📝 License

Vedi [`LICENSE`](LICENSE) file.

## 🎓 References

- **Dataset**: GUIDE (Guided Response Dataset) - Microsoft Security AI Research
- **MITRE ATT&CK**: https://attack.mitre.org/
- **Bayesian Smoothing**: Wilson score interval per binary data
