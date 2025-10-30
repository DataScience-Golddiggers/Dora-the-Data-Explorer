# Quick Start Guide - Dataset GUIDE

## Esempio d'uso del modulo guide_utils.py

```python
# Importa il modulo personalizzato
from guide_utils import (
    load_guide_dataset,
    clean_incident_grade,
    extract_temporal_features,
    parse_mitre_techniques,
    create_aggregated_features,
    full_preprocessing_pipeline
)

# Opzione 1: Pipeline completo (raccomandato)
df = full_preprocessing_pipeline('../data/GUIDE_Train.csv')

# Opzione 2: Step by step
df = load_guide_dataset('../data/GUIDE_Train.csv', sample_frac=0.1)  # Carica 10%
df = clean_incident_grade(df)
df = extract_temporal_features(df)
df = parse_mitre_techniques(df)
df = create_aggregated_features(df)

# Analizza la distribuzione del target
from guide_utils import get_incident_grade_distribution
dist = get_incident_grade_distribution(df)

# Analizza cardinalità features
from guide_utils import get_top_features_by_cardinality
card = get_top_features_by_cardinality(df, n=15)

# Prepara per modeling
from guide_utils import prepare_for_modeling
X, y = prepare_for_modeling(df, target_col='IncidentGrade')
```

## Struttura Cartelle Raccomandata

```
EDA/
├── data/
│   ├── GUIDE_Train.csv       # Dataset training
│   └── GUIDE_Test.csv        # Dataset test
├── notebook/
│   ├── EDA-Cybersec.ipynb    # Notebook principale EDA
│   ├── guide_utils.py        # Funzioni di supporto
│   ├── README_GUIDE.md       # Documentazione dataset
│   └── QUICK_START.md        # Questa guida
└── models/                   # (da creare) Modelli salvati
```

## Workflow Consigliato

### 1. Esplorazione Iniziale
Apri `EDA-Cybersec.ipynb` ed esegui le celle per:
- Caricare il dataset
- Visualizzare statistiche descrittive
- Analizzare valori mancanti
- Esplorare distribuzioni

### 2. Feature Engineering
Usa `guide_utils.py` per:
- Estrarre feature temporali
- Parsare MITRE Techniques
- Creare aggregazioni
- Gestire missing values

### 3. Analisi Approfondite
Nel notebook, analizza:
- Pattern temporali
- Distribuzione per categoria
- Relazioni con il target
- Co-occorrenze alert/incident

### 4. Preparazione Modeling
```python
from guide_utils import prepare_for_modeling

# Prepara i dati
X, y = prepare_for_modeling(df)

# Gestisci categorie
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.3, 
    stratify=y_encoded,
    random_state=42
)
```

## Metriche di Valutazione

Per il dataset GUIDE, Microsoft raccomanda:

### Metrica Principale
- **Macro-F1 Score**: Media non pesata degli F1-score per classe
  ```python
  from sklearn.metrics import f1_score
  f1_macro = f1_score(y_test, y_pred, average='macro')
  ```

### Metriche Secondarie
- **Precision** e **Recall** per classe
- **Confusion Matrix**
- **ROC-AUC** (one-vs-rest per multiclass)

```python
from sklearn.metrics import classification_report, confusion_matrix

# Report completo
print(classification_report(y_test, y_pred, 
                          target_names=['BenignPositive', 'FalsePositive', 'TruePositive']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
```

## Tips & Best Practices

### Performance
- Il dataset è grande (~9.5M righe): considera sampling per test rapidi
- Usa `dtype` ottimizzati per risparmiare memoria
- Considera Dask/Modin per dataset ancora più grandi

### Feature Engineering
- Le colonne numeriche sono ID anonimizzati → usa come categorie
- MitreTechniques può contenere multiple tecniche → parsale
- Sfrutta la struttura gerarchica (Evidence → Alert → Incident)

### Modeling
- Dataset sbilanciato → usa stratified sampling
- Molte categorie high-cardinality → target encoding o embedding
- Cross-validation con attenzione a data leakage

### Missing Values
- ActionGrouped/ActionGranular: ~99% missing (task secondario)
- MitreTechniques: ~57% missing (crea "Unknown")
- SuspicionLevel/LastVerdict: ~80% missing (indicatori binari)

## Troubleshooting

### Memoria insufficiente
```python
# Carica un campione
df = load_guide_dataset('../data/GUIDE_Train.csv', sample_frac=0.1)

# Oppure usa chunking
chunks = pd.read_csv('../data/GUIDE_Train.csv', chunksize=100000)
for chunk in chunks:
    process_chunk(chunk)
```

### Encoding categorici
```python
# Per categorie con pochi valori
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Category_Encoded'] = le.fit_transform(df['Category'])

# Per high-cardinality
from category_encoders import TargetEncoder
te = TargetEncoder()
df_encoded = te.fit_transform(df[['OrgId', 'DetectorId']], y)
```

### Gestione timestamp
```python
# Se Timestamp non viene parsato correttamente
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%dT%H:%M:%S.%fZ')
```

## Risorse Aggiuntive

- **Dataset**: [Kaggle Competition](https://www.kaggle.com/)
- **Paper**: GUIDE on arXiv
- **MITRE ATT&CK**: https://attack.mitre.org/
- **Documentazione Microsoft**: Copilot for Security

---
*Per domande o problemi, consulta README_GUIDE.md*

