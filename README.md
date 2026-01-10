
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


Binary cybersecurity incident classification using Microsoft's **GUIDE dataset**. Predicts `BinaryIncidentGrade` (0=Non-TP, 1=TP) from hierarchical security evidence.

## 🎯 Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|----------|-----------|--------|---------------|---------|
| **XGBoost** | **0.8019** | **0.8274** | **0.7888** | **0.8076** | **0.9061** |
| Random Forest v2 | 0.7919 | 0.8252 | 0.7681 | 0.7956 | 0.8992 |
| Decision Tree | 0.7868 | 0.8077 | 0.7819 | 0.7946 | 0.8854 |
| MLP (Sklearn) | 0.7679 | 0.8101 | 0.7311 | 0.7686 | 0.8720 |

**Dataset Stats**: 9.5M evidence records → 1.6M alerts → 1M incidents | 441 MITRE ATT&CK techniques | 33 entity types

## 📁 Project Structure

```
notebook/          # All analysis & modeling code (PRIMARY LOCATION)
├── guide_utils.py              # Core preprocessing utilities
├── 1-Advanced_EDA.ipynb        # Initial exploration & stats
├── 2-FeatureEngineering.ipynb   # 23 aggregated features & Smoothed Risk
├── 3-FeatureEngineering_Pipeline.ipynb  # Pipeline with anti-leakage split
├── 4-Model_Training_and_Comparison.ipynb # Tuning (CV) & Real-world eval
├── exploration/
│   ├── analisi_mitre_preprocessing.ipynb  # MITRE one-hot encoding
│   └── Initial_EDA.ipynb
└── tests/                       # Model-specific development & tests

models/            # Trained models & metrics (Jan 2026 Revision)
├── xgboost_v2/           # model.pkl, metrics.json, importance
├── random_forest_v2/
├── decision_tree/
└── mlp_baseline/

docs/              # Documentation
├── methodology.md       # Design decisions & Rationale
├── CHANGELOG.md         # Critical revision details
├── PIPELINE_USAGE.md    # Preprocessing guide
└── README_GUIDE.md      # Dataset reference
```

## 🚀 Quick Start

1. Install requirements: `pip install -r requirements.txt`
2. Run notebooks in `notebook/` in order (1 to 4).
3. The final model comparison and evaluation is in `4-Model_Training_and_Comparison.ipynb`.


## 📊 Feature Engineering

**Hierarchical aggregation** (Evidence → Incident level):
- **Target**: `max` (Incident is TP if any evidence is TP)
- **Aggregations**: `nunique` for Alerts/Countries, `count` for evidences, `mean/max` for Risk scores.
- **Temporal**: `Duration_seconds`, `Hour_mean`, `IsWeekend`.

**Advanced Encoding**:
- **Smoothed Risk Score**: Bayesian Target Encoding on `AlertTitle` with smoothing (α=5) to prevent overfitting on rare categories.
- **Frequency Encoding**: Applied to high-cardinality categorical features (`GeoLoc`, `EntityType`).

**MITRE processing**:
- Parse semicolon-separated techniques.
- **Top 20 techniques** (by frequency) → one-hot encoded.
- `MitreCount` column for total techniques per incident.

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
- **Parameters**: `max_depth=10`, `learning_rate=0.1`, `n_estimators=300`, `subsample=0.9`
- **Evaluation**: 5-fold stratified CV (Best CV F1: 0.8679)
- **Top Features** (from `feature_importance.csv`):
1. `SmoothedRisk_avg` (0.30 gain)
2. `T1078_sum` (0.12 gain)
3. `EvidenceRole_Related_sum` (0.06 gain)
4. `GeoLoc_freq_avg` (0.05 gain)
5. `EntityType_freq_mean` (0.04 gain)

### Random Forest v2 (`models/random_forest_v2/`)
- `n_estimators=150`, `max_depth=20`, `min_samples_split=5`
- F1 Score (TP): 0.7956

### Decision Tree (`models/decision_tree/`)
- `max_depth=15`, `min_samples_split=20`
- F1 Score (TP): 0.7946

### MLP (`models/mlp_baseline/`)
- **Sklearn**: 2 hidden layers (128-64), ReLU, Early Stopping.
- **PyTorch**: 3 layers (128-64-32), Dropout, BatchNorm (F1: 0.7651).
- F1 Score (TP): 0.7686 (Sklearn)

## 🎓 Key Insights

1. **Validation Strategy**: Evaluation is performed on an **imbalanced (real-world) distribution** to provide honest performance estimates.
2. **Anti-Leakage Split**: Data is split at the **Incident level**, ensuring all evidences of a single incident stay within the same fold (Train or Val/Test).
3. **Hyperparameter Tuning**: All models tuned using **Stratified 5-fold Cross-Validation** via `RandomizedSearchCV`.
4. **Missing Values**:
- `MitreTechniques` → Top 20 OHE + Count.
- `SuspicionLevel` → `SuspicionLevel_IsMissing` indicator.
- Use median/mode imputation or constant indicators.

## 📖 Documentation

- **[PIPELINE_USAGE.md](docs/PIPELINE_USAGE.md)** - Step-by-step preprocessing guide
- **[QUICK_START.md](docs/QUICK_START.md)** - Setup & first experiments
- **[README_GUIDE.md](docs/README_GUIDE.md)** - GUIDE dataset reference
- **[PIPELINE_SUMMARY.md](docs/PIPELINE_SUMMARY.md)** - Architecture overview

## 🔍 Evaluation

**Primary Metric**: F1 Score (TP) and ROC AUC

```python
from sklearn.metrics import classification_report, roc_auc_score

print(classification_report(y_test, y_pred, target_names=['Non-TP', 'TP']))
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba)}")
```

**Performance Analysis** (XGBoost v2):
- **ROC AUC**: 0.9061 (Excellent discrimination between threats and noise)
- **Recall**: 0.79 (Detects ~79% of actual threats)
- **Precision**: 0.83 (When it alerts, it's correct 83% of the time)

## 🤝 Contributing

When adding features:
1. Aggregate to Incident level in `2-FeatureEngineering.ipynb`
2. Update `guide_utils.create_aggregated_features()`
3. Re-run balancement and model training notebooks
4. Compare metrics in `9-ModelComparison.ipynb`

## 📄 License

Dataset: Microsoft GUIDE (Guided Response Dataset) - Public research dataset

</p>
