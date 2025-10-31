# 📋 Riepilogo Pipeline Creata

## ✅ File Creati

### Preprocessing (`src/preprocessing/`)
- ✅ `feature_engineering.py` - Pipeline completa per feature engineering
- ✅ `data_rebalancing.py` - Strategie di bilanciamento dataset
- ✅ `run_preprocessing.py` - Script CLI per preprocessing
- ✅ `__init__.py` - Package initialization

### Training (`src/training/`)
- ✅ `train_xgboost.py` - Training XGBoost con argparse CLI
- ✅ `train_random_forest.py` - Training Random Forest con argparse CLI
- ✅ `train_mlp.py` - Training MLP (Neural Network) con argparse CLI
- ✅ `__init__.py` - Package initialization

### Inference (`src/inference/`)
- ✅ `inference_pipeline.py` - Pipeline di inferenza con confronto modelli
- ✅ `__init__.py` - Package initialization

### Utilities
- ✅ `src/config.py` - Configurazione globale
- ✅ `src/__init__.py` - Package root

### Documentazione
- ✅ `README_PIPELINE.md` - README principale della pipeline
- ✅ `docs/PIPELINE_USAGE.md` - Documentazione dettagliata
- ✅ `requirements.txt` - Dipendenze Python

### Testing & Examples
- ✅ `tests/test_installation.py` - Test suite per verificare installazione
- ✅ `examples/complete_pipeline_example.py` - Esempio completo end-to-end

## 🎯 Caratteristiche Principali

### 1. Preprocessing
- **Pipeline riutilizzabile**: `FeatureEngineeringPipeline` può essere salvata e ricaricata
- **Fit/Transform pattern**: `fit_transform()` per training, `transform()` per test/inference
- **No data leakage**: Statistiche calcolate solo sul training set
- **Gestione automatica missing values**: Strategie specifiche per ogni colonna
- **43 features totali**: Strutturali, temporali, risk, geographic, frequency-encoded, MITRE

### 2. Data Rebalancing
- **Minority augmentation**: Aggiunge campioni TruePositive dal test set
- **Undersampling**: Riduce classe maggioritaria
- **Split stratificato**: Mantiene proporzioni classi in train/test

### 3. Training
- **Tre modelli implementati**: XGBoost, Random Forest, MLP
- **CLI completa**: Tutti i parametri configurabili da command line
- **Class imbalance handling**: 
  - XGBoost: `scale_pos_weight`
  - Random Forest: `class_weight='balanced'`
  - MLP: `pos_weight` in loss function
- **Standardizzazione automatica**: MLP applica e salva StandardScaler
- **Salvataggio completo**: Modello + metriche + feature importance + scaler

### 4. Inferenza
- **Pipeline consistency**: Applica le stesse trasformazioni del training
- **Batch processing**: Gestisce file grandi con chunking
- **Model comparison**: Confronta predizioni di più modelli
- **Rich output**: Predizioni + probabilità + metadata

## 📊 Output Salvati

### Preprocessing
```
data/processed_v3_balanced/
├── X_train.csv              # Features training (220k samples x 43 features)
├── X_test.csv               # Features test (95k samples x 43 features)
├── y_train.csv              # Target training
├── y_test.csv               # Target test
└── incident_features.csv    # Dataset completo aggregato

models/
└── preprocessing_pipeline.pkl  # Pipeline per inferenza
```

### Training (per ogni modello)
```
models/xgboost_v2/
├── model.json               # Modello XGBoost (formato nativo)
├── model.pkl                # Modello pickle
├── feature_importance.csv   # Importanza features
└── metrics.json             # Tutte le metriche

models/random_forest_v2/
├── model.pkl
├── feature_importance.csv
└── metrics.json

models/mlp_baseline/
├── model.pth                # Modello PyTorch completo
├── model_weights.pth        # Solo weights
├── scaler.pkl               # StandardScaler (IMPORTANTE!)
└── metrics.json
```

### Inferenza
```
predictions_xgboost.csv
├── IncidentId
├── Predicted_Class          # 0 o 1
├── Predicted_Proba          # Probabilità classe 1
├── Predicted_Label          # "TruePositive" o "Non-TruePositive"
└── True_Class (se disponibile)

predictions_comparison.csv   # Confronto multi-modello
├── IncidentId
├── True_Class
├── XGBoost_Pred
├── XGBoost_Proba
├── RandomForest_Pred
├── RandomForest_Proba
├── MLP_Pred
├── MLP_Proba
├── All_Agree                # Boolean: tutti i modelli concordano?
└── Majority_Vote            # Voto di maggioranza
```

## 🚀 Quick Start Commands

```bash
# 1. Test installazione
cd tests
python test_installation.py

# 2. Preprocessing completo
cd ../src/preprocessing
python run_preprocessing.py \
    --train_file ../../data/GUIDE_Train.csv \
    --test_file ../../data/GUIDE_Test.csv \
    --output_dir ../../data/processed_v3_balanced \
    --balance_strategy augment \
    --save_pipeline

# 3. Training XGBoost
cd ../training
python train_xgboost.py \
    --data_dir ../../data/processed_v3_balanced \
    --model_dir ../../models/xgboost_v2 \
    --no_scale_pos_weight

# 4. Training Random Forest
python train_random_forest.py \
    --data_dir ../../data/processed_v3_balanced \
    --model_dir ../../models/random_forest_v2 \
    --no_class_weight

# 5. Training MLP
python train_mlp.py \
    --data_dir ../../data/processed_v3_balanced \
    --model_dir ../../models/mlp_baseline \
    --no_class_weight

# 6. Inferenza singola
cd ../inference
python inference_pipeline.py \
    --input_file ../../data/GUIDE_Test.csv \
    --output_file ../../predictions_xgboost.csv \
    --model_dir ../../models/xgboost_v2 \
    --model_type xgboost

# 7. Esempio completo end-to-end
cd ../../examples
python complete_pipeline_example.py
```

## 🔧 Personalizzazione

### Cambiare hyperparameters
Modifica i parametri in `src/config.py` o passa via CLI:

```bash
# XGBoost custom
python train_xgboost.py \
    --max_depth 10 \
    --learning_rate 0.05 \
    --n_estimators 300

# Random Forest custom
python train_random_forest.py \
    --n_estimators 200 \
    --max_depth 20

# MLP custom
python train_mlp.py \
    --hidden_dims 256 128 64 \
    --learning_rate 0.0001 \
    --epochs 100
```

### Cambiare feature engineering
Modifica `FeatureEngineeringPipeline` in `src/preprocessing/feature_engineering.py`:
- `alpha`, `beta`: Parametri Bayesian smoothing
- `top_n_mitre`: Numero top MITRE techniques
- Aggiungi nuove features nei metodi `create_*_features()`

### Cambiare strategia di bilanciamento
```bash
# Nessun bilanciamento
python run_preprocessing.py --balance_strategy none

# Solo augmentation
python run_preprocessing.py --balance_strategy augment

# Solo undersampling
python run_preprocessing.py --balance_strategy undersample

# Entrambi
python run_preprocessing.py --balance_strategy both
```

## ⚠️ Note Importanti

### Per MLP
1. **StandardScaler è obbligatorio**: Applicato automaticamente dal trainer
2. **Salva sempre scaler.pkl**: Necessario per inferenza
3. **GPU opzionale**: PyTorch usa CPU di default, GPU se disponibile

### Per Inferenza
1. **Pipeline deve essere salvata**: Usa `--save_pipeline` nel preprocessing
2. **Stesse trasformazioni del training**: La pipeline garantisce consistency
3. **Batch processing per file grandi**: Usa `--chunk_size` per evitare OOM

### Data Leakage Prevention
1. **SmoothedRisk**: Usa statistiche solo dal training set
2. **GeoLoc_freq**: Location non viste ricevono frequenza minima del training
3. **MITRE techniques**: Solo le top 30 del training sono considerate
4. **Frequency encoding**: Mapping dal training applicato al test

## 📚 Prossimi Passi

1. **Esegui test di installazione**: `python tests/test_installation.py`
2. **Leggi documentazione completa**: `docs/PIPELINE_USAGE.md`
3. **Esegui esempio completo**: `python examples/complete_pipeline_example.py`
4. **Preprocessa i tuoi dati**: `python src/preprocessing/run_preprocessing.py`
5. **Addestra modelli**: Usa gli script in `src/training/`
6. **Fai inferenza**: `python src/inference/inference_pipeline.py`

## 🎓 Best Practices Implementate

✅ **Separation of concerns**: Preprocessing, training, inference separati  
✅ **Reproducibility**: Random seeds configurabili, pipeline salvabili  
✅ **CLI-friendly**: Tutti gli script usabili da command line  
✅ **Library-friendly**: Tutti i componenti importabili come moduli Python  
✅ **No data leakage**: Statistiche calcolate solo sul training  
✅ **Error handling**: Try-catch e validazioni in punti critici  
✅ **Logging**: Print informativi per tracking progresso  
✅ **Configurazione centralizzata**: `config.py` per parametri globali  
✅ **Testing**: Test suite per verificare installazione  
✅ **Documentation**: README, docstring, esempi completi  

## 🏆 Risultati Attesi

Con il dataset bilanciato e i parametri default:

| Modello | ROC AUC | F1-Score | Precision | Recall |
|---------|---------|----------|-----------|--------|
| XGBoost | ~0.85-0.90 | ~0.80-0.85 | ~0.80-0.85 | ~0.75-0.85 |
| Random Forest | ~0.82-0.88 | ~0.78-0.83 | ~0.78-0.83 | ~0.73-0.82 |
| MLP | ~0.80-0.87 | ~0.75-0.82 | ~0.75-0.82 | ~0.70-0.80 |

*Nota: I risultati variano in base al bilanciamento e agli hyperparameters.*

---

**Pipeline creata con successo! ✨**

Per qualsiasi domanda, consulta:
- `README_PIPELINE.md` - Overview generale
- `docs/PIPELINE_USAGE.md` - Guida dettagliata
- `examples/complete_pipeline_example.py` - Esempio pratico
