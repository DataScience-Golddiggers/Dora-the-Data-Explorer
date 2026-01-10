
# Dora the Data Explorer 🔍

<p align="center">

**Data Science - Exploration and Classification Project**

[![Python](https://img.shields.io/badge/Python-3.13-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
[![Huggingface](https://img.shields.io/badge/-HuggingFace-FDEE21?style=for-the-badge&logo=HuggingFace&logoColor=black)](https://huggingface.co)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
![PyCharm](https://img.shields.io/badge/PyCharm-000000.svg?&style=for-the-badge&logo=PyCharm&logoColor=white)
![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)
![macOS](https://img.shields.io/badge/mac%20os-000000?style=for-the-badge&logo=apple&logoColor=white)
[![License](https://img.shields.io/badge/MIT-green?style=for-the-badge)](LICENSE)

<p align="center">
<img src="docs\public\rdm1.png" alt="DDE" width="100%">
</p>


Multi-class cybersecurity incident classification using Microsoft's **GUIDE dataset**. Predicts `IncidentGrade` (TruePositive, BenignPositive, FalsePositive) from hierarchical security evidence.

## 🎯 Performance

| Model | Macro F1 | TP F1 | BP F1 | FP F1 |
|-------|----------|-------|-------|-------|
| **XGBoost v2** | **0.8559** | 0.87 | 0.89 | 0.81 |
| Random Forest v2 | 0.8234 | 0.84 | 0.86 | 0.77 |
| Decision Tree | 0.7891 | 0.81 | 0.83 | 0.73 |
| MLP Baseline | 0.7456 | 0.76 | 0.80 | 0.68 |

**Dataset Stats**: 9.5M evidence records → 1.6M alerts → 1M incidents | 441 MITRE ATT&CK techniques | 33 entity types

## 📁 Project Structure

```
notebook/          # All analysis & modeling code (PRIMARY LOCATION)
├── guide_utils.py              # Core preprocessing utilities
├── Test_00-Advanced_EDA.ipynb  # Initial exploration
├── Test_03-FeatureEngineering_v3.ipynb  # 23 aggregated features
├── exploration/
│   ├── analisi_mitre_preprocessing.ipynb  # MITRE one-hot encoding
│   └── Initial_EDA.ipynb
├── Test_03-XGBoost_v2_Model.ipynb  # Best model (0.8559 F1)
├── Test_02-RandomForest.ipynb
├── Test_09-DecisionTree.ipynb
├── Test_07-NeuralNetwork_MLP.ipynb
└── Test_14_DataRebalancing_Hybrid.ipynb  # Class balancing experiments

models/            # Trained models & metrics
├── xgboost_v2/           # model.json, feature_importance.csv, metrics.json
├── random_forest_v2/
├── decision_tree/
└── mlp_baseline/

data/
├── processed/     # X_train.csv, X_test.csv, y_train.csv, y_test.csv
└── GUIDE_Train.csv  # Raw dataset (not in repo)

docs/              # Documentation
├── PIPELINE_USAGE.md    # Full preprocessing pipeline guide
├── QUICK_START.md       # Setup & first experiments
└── README_GUIDE.md      # Dataset documentation
```

## 🚀 Quick Start

Only start jupiter notebooks in the displayed order, from 1-... to 9-..., then wait and enjoy!


## 📊 Feature Engineering

**Hierarchical aggregation** (Evidence → Incident level):
```python
# Implemented in guide_utils.create_aggregated_features()
incident_features = evidence_df.groupby('IncidentId').agg({
    'AlertId': 'nunique',           # NumAlerts
    'EntityType': 'nunique',        # NumEntityTypes
    'SuspicionLevel': lambda x: x.notna().sum(),  # NumWithSuspicion
    'CountryCode': 'nunique',       # NumCountries
    'Timestamp': ['min', 'max']     # Duration calculation
})
```

**Temporal extraction** (from `Timestamp`):
- `Hour_First`, `Hour_Last`, `Hour_Avg` (0-23)
- `DayOfWeek` (0-6), `IsWeekend` (binary)
- `Duration_seconds` (time between first/last evidence)

**MITRE processing** (from `MitreTechniques` column):
- Parse semicolon-separated lists: `"T1078;T1566"` → separate columns
- 30 most frequent techniques (>0.5% occurrence) → one-hot encoded
- `n_rare` column for rare techniques count

## 🛠️ Utilities (`guide_utils.py`)

| Function | Purpose |
|----------|---------|
| `load_guide_dataset(path, sample_frac=0.1)` | Load with memory-efficient sampling |
| `full_preprocessing_pipeline(path)` | Raw CSV → modeling-ready X, y |
| `extract_temporal_features(df)` | Hour, DayOfWeek, IsWeekend, Duration |
| `parse_mitre_techniques(df)` | Count & indicator for MITRE codes |
| `create_aggregated_features(df)` | Evidence→Incident aggregations |
| `prepare_for_modeling(df, target_col)` | Split X/y, drop IDs, stratified split |

## 📈 Model Specifications

### XGBoost v2 (Best - `models/xgboost_v2/`)
- **Parameters**: `max_depth=6`, `learning_rate=0.1`, `n_estimators=200`
- **Evaluation**: 5-fold stratified CV
- **Top Features** (from `feature_importance.csv`):
1. `NumAlerts` (0.18 gain)
2. `NumEvidences` (0.14 gain)
3. `Duration_seconds` (0.11 gain)
4. `NumEntityTypes` (0.09 gain)
5. `Hour_First` (0.08 gain)

### Random Forest v2 (`models/random_forest_v2/`)
- 200 trees, `max_depth=20`, `min_samples_split=10`
- Macro F1: 0.8234

### Decision Tree (`models/decision_tree/`)
- `max_depth=15`, Gini impurity
- Macro F1: 0.7891

### MLP (`models/mlp_baseline/`)
- 3 hidden layers (128-64-32), ReLU, Dropout(0.3)
- Macro F1: 0.7456

## 🎓 Key Insights

1. **Class Imbalance Strategy**: manual balancement, undersampling and SMOTE
2. **ID Columns**: Treat `DeviceId`, `AlertId`, `OrgId` by deleting them
3. **Missing Values**:
- `MitreTechniques` (57% missing) → `HasMitreTechniques` binary indicator
- `SuspicionLevel` (80% missing) → `SuspicionLevel_IsMissing` indicator
- Use `-999` or 'missing' for NaN 
4. **Never model at Evidence level**: Always aggregate to Incident before classification

## 📖 Documentation

- **[PIPELINE_USAGE.md](docs/PIPELINE_USAGE.md)** - Step-by-step preprocessing guide
- **[QUICK_START.md](docs/QUICK_START.md)** - Setup & first experiments
- **[README_GUIDE.md](docs/README_GUIDE.md)** - GUIDE dataset reference
- **[PIPELINE_SUMMARY.md](docs/PIPELINE_SUMMARY.md)** - Architecture overview

## 🔍 Evaluation

**Primary Metric**: Macro F1-Score (equal weight for TP/BP/FP)

```python
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred, 
    target_names=['BenignPositive', 'FalsePositive', 'TruePositive']))
```

**Confusion Matrix Analysis** (XGBoost v2):
- Strongest: TruePositive (87% F1) - correctly identifies real threats
- Weakest: FalsePositive (81% F1) - some confusion with BenignPositive

## 🤝 Contributing

When adding features:
1. Aggregate to Incident level in `2-FeatureEngineering.ipynb`
2. Update `guide_utils.create_aggregated_features()`
3. Re-run balancement and model training notebooks
4. Compare metrics in `9-ModelComparison.ipynb`

## 📄 License

Dataset: Microsoft GUIDE (Guided Response Dataset) - Public research dataset

</p>
