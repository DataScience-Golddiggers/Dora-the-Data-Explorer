# AI Coding Agent Instructions - Dora the Data Explorer

## Project Overview
Cybersecurity incident classification using the **GUIDE dataset** (Microsoft's largest public cybersecurity incident dataset). This is a multi-class classification problem predicting `IncidentGrade` (TruePositive, BenignPositive, FalsePositive) from security alerts and evidence.

**Key Stats**: ~9.5M evidence records, 1M incidents, 441 MITRE ATT&CK techniques, 33 entity types  
**Primary Metric**: Macro F1-Score (equal weight for all three classes)  
**Best Model Performance**: XGBoost with 0.8559 macro F1-score

## Architecture & Data Flow

### Hierarchical Structure (Critical for Feature Engineering)
```
Evidence (9.5M rows) → Alert (1.6M) → Incident (1M)
```
All features must be **aggregated to Incident level** for classification. Never model at Evidence or Alert level directly.

### Pipeline Stages
1. **EDA** (`01 - EDA-Cybersec.ipynb`) - Initial exploration, missing values analysis
2. **Feature Engineering** (`02 - FeatureEngineering.ipynb`) - Evidence→Incident aggregation, creates 23 features
3. **MITRE Preprocessing** (`analisi_mitre_preprocessing.ipynb`) - One-hot encoding of 25 most frequent MITRE techniques (>0.5% occurrence threshold)
4. **Modeling** (`03 - XGBaby.ipynb`) - XGBoost training, hyperparameter tuning, evaluation

### Processed Data Structure (`data/processed/`)
- `X_train.csv` / `X_test.csv`: 23 aggregated features (448,901 incidents for train)
- `y_train.csv` / `y_test.csv`: Target labels (IncidentGrade)
- `incident_features.csv`: Full incident-level aggregated dataset
- `label_encoders.pkl`: Encoder mappings for categorical variables

## Critical Project Conventions

### Data Handling
- **ID columns are anonymized**: Treat numeric columns like `DeviceId`, `AlertId`, `OrgId` as categorical, NOT continuous
- **Missing value strategy by column**:
  - `MitreTechniques` (57% missing) → Create 'unknown' category + binary indicator `HasMitreTechniques`
  - `SuspicionLevel` / `LastVerdict` (~80% missing) → Binary indicators (`SuspicionLevel_IsMissing`)
  - `ActionGrouped` / `ActionGranular` (99% missing) → Drop or use for secondary task only
- **Use -999 for NaN in modeling**: XGBoost handles this pattern well (see `02 - FeatureEngineering.ipynb` cell with `fillna(-999)`)

### Feature Engineering Patterns (see `guide_utils.py`)
Always aggregate from Evidence level using these patterns:
```python
# Structural counts per incident
NumAlerts = df.groupby('IncidentId')['AlertId'].nunique()
NumEvidences = df.groupby('IncidentId')['Id'].count()
NumEntityTypes = df.groupby('IncidentId')['EntityType'].nunique()

# Temporal features (extract from Timestamp)
Hour_First, Hour_Last, Hour_Avg, DayOfWeek, IsWeekend, Duration_seconds

# Geographic diversity
NumCountries, NumStates, NumCities

# Security indicators
NumWithSuspicion = df.groupby('IncidentId')['SuspicionLevel'].apply(lambda x: x.notna().sum())
NumUniqueMitre = count unique MITRE techniques per incident
```

### MITRE Techniques Processing
- Parse semicolon-separated lists: `T1078;T1566;T1098` → separate columns
- Normalize codes: Strip whitespace, ensure 'T' prefix, take base code (before `.`)
- Filter by frequency: Only encode techniques appearing in >0.5% of incidents (25 techniques total)
- Create `n_rare` column for rare techniques count

### Stratified Splitting (MANDATORY)
Dataset is imbalanced (BP: 43%, TP: 35%, FP: 21%). Always use:
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
```

### Evaluation Metrics Priority
1. **Macro F1-Score** (primary, equal class importance)
2. Per-class Precision/Recall/F1 (identify weak classes)
3. Confusion Matrix (understand TP vs BP vs FP confusion patterns)
4. Accuracy (secondary, less important due to imbalance)

## Key Utilities (`notebook/guide_utils.py`)

**Use these instead of writing from scratch**:
- `load_guide_dataset(path, sample_frac=0.1)` - Load with optional sampling for memory
- `full_preprocessing_pipeline(path)` - Complete end-to-end preprocessing
- `extract_temporal_features(df)` - Hour, DayOfWeek, IsWeekend, TimeOfDay
- `parse_mitre_techniques(df)` - Count techniques, create binary indicators
- `create_aggregated_features(df)` - Evidence→Incident aggregations
- `prepare_for_modeling(df, target_col='IncidentGrade')` - Splits X and y, drops IDs

## Modeling Best Practices

### XGBoost Configuration (Current Best)
```python
xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    eval_metric='mlogloss',
    scale_pos_weight=[...],  # Handle class imbalance
    tree_method='hist',  # Faster for large datasets
    n_jobs=-1
)
```

### Hyperparameters Tested (see `03 - XGBaby.ipynb`)
- `max_depth`: 6-10 (default 6 works well)
- `learning_rate`: 0.1-0.3
- `n_estimators`: 100-300
- `min_child_weight`: 1-5
- Cross-validate with `StratifiedKFold(n_splits=5)`

### Feature Importance Analysis
Model saves to `models/feature_importance.csv`. Top contributors:
- Structural: `NumAlerts`, `NumEvidences`, `NumEntityTypes`
- Temporal: `Hour_First`, `Duration_seconds`
- Security: `NumWithSuspicion`, `NumUniqueMitre`

## Common Workflows

### Quick EDA on New Data Slice
```python
from guide_utils import load_guide_dataset, get_incident_grade_distribution
df = load_guide_dataset('data/GUIDE_Train.csv', sample_frac=0.01)
get_incident_grade_distribution(df)
```

### Add New Aggregated Feature
```python
# In 02 - FeatureEngineering.ipynb, add to aggregation dict:
incident_agg = df.groupby('IncidentId').agg({
    'YourColumn': 'nunique',  # or 'count', 'mean', lambda x: ...
}).reset_index()
```

### Evaluate Model Changes
Always report macro F1 + per-class breakdown:
```python
from sklearn.metrics import f1_score, classification_report
macro_f1 = f1_score(y_test, y_pred, average='macro')
print(classification_report(y_test, y_pred, 
      target_names=['BenignPositive', 'FalsePositive', 'TruePositive']))
```

## Debugging & Performance

### Memory Issues
- Use `sample_frac=0.1` when iterating on code
- Large dataset: 9.5M rows × 45 columns ≈ 2-3 GB
- Processed features: 450K incidents × 23 features ≈ 80 MB

### Data Validation Checks
- `IncidentGrade` never null after cleaning (target required)
- All numeric IDs should be treated as `category` dtype
- Timestamp must parse: `pd.to_datetime(df['Timestamp'])`
- MITRE techniques: Check for malformed entries (missing 'T', extra dots)

## File Locations & Naming
- Raw data: `data/GUIDE_Train.csv`, `data/GUIDE_Test.csv`
- Processed: `data/processed/*.csv`
- Models: `models/xgboost_optimized.json`
- Notebooks numbered sequentially: `01`, `02`, `03` prefix
- Utility functions: `notebook/guide_utils.py` (import via `from guide_utils import ...`)

## References
- Dataset: GUIDE (Guided Response Dataset) - Microsoft Security AI Research
- MITRE ATT&CK Framework: https://attack.mitre.org/
- Challenge metric: Macro F1-Score (equal weight for TP, BP, FP)
