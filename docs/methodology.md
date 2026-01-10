# Methodology & Design Decisions

Questo documento descrive le scelte metodologiche adottate nel progetto di classificazione degli incidenti di sicurezza informatica utilizzando il dataset GUIDE di Microsoft.

---

## 1. Formulazione del Problema

### Classificazione Binaria vs Ternaria

**Scelta:** Classificazione binaria (TruePositive vs Altri)

**Motivazione:**
- Il dataset GUIDE originale prevede 3 classi: `TruePositive`, `BenignPositive`, `FalsePositive`
- Abbiamo scelto la formulazione binaria perché:
  1. **Interpretabilità**: La domanda "È una minaccia reale?" è più immediata per gli analisti SOC
  2. **Semplicità**: Permette l'uso di metriche standard (ROC-AUC, F1 binario) senza ambiguità
  3. **Focus**: Massimizza la detection delle minacce reali, l'obiettivo primario in cybersecurity

**Trade-off:** Si perde la distinzione tra BenignPositive (falso allarme benigno) e FalsePositive (errore del sistema), che potrebbero richiedere azioni diverse.

---

## 2. Strategia di Splitting dei Dati

### Split a Livello Incident (Anti-Leakage)

**Problema:** Nel dataset GUIDE, ogni incidente ha multiple evidenze (righe). Uno split casuale a livello riga potrebbe inserire evidenze dello stesso incidente sia in train che in validation, causando data leakage.

**Soluzione implementata:**
```
1. Estrazione lista unica IncidentId con rispettivo Target
2. Split stratificato degli IncidentId (90% train, 10% validation)
3. Assegnazione di TUTTE le evidenze di ogni incidente al rispettivo fold
```

**Codice di riferimento:** Notebook 14, Sezione 2

### Riallocazione Test Set → Training

**Contesto:** Il dataset GUIDE_Test originale contiene ~509k incidenti, un numero eccessivo per un semplice hold-out set.

**Decisione:** Abbiamo mantenuto 10k incidenti per classe nel test finale (20k totali) e riallocato il resto (~489k incidenti) al training.

**Giustificazione:**
- 20k campioni bilanciati sono sufficienti per una valutazione statisticamente robusta
- Aumentare il training set migliora la generalizzazione dei modelli
- Non partecipiamo a una competizione, quindi non dobbiamo mantenere il test set originale

**Nota:** Questa scelta deve essere documentata in qualsiasi paper/report per riproducibilità.

---

## 3. Feature Engineering

### 3.1 Target Encoding (Smoothed Risk Score)

**Tecnica:** Bayesian Smoothed Target Encoding su `AlertTitle`

**Formula:**
```
smoothed_score = (count * mean + α * global_mean) / (count + α)
```
dove α = 5 (prior weight)

**Perché smoothing?**
- Categorie rare con poche osservazioni tendono ad avere medie estreme (0 o 1)
- Lo smoothing "tira" le medie verso la media globale, riducendo overfitting
- α = 5 è un valore conservativo che bilancia informazione locale vs globale

**Anti-leakage:** La mappa è calcolata SOLO su train_fold e applicata a val/test.

### 3.2 Frequency Encoding

**Colonne:** `GeoLoc`, `DeviceName`, `FileName`, `IpAddress`, `Url`

**Limitazione nota:** `FileName`, `IpAddress`, `Url` hanno cardinalità molto alta. La maggior parte dei valori appare una sola volta, risultando in frequenze ~0.

**Mitigazione:** La feature importance analysis (Notebook 15, Sezione 5) mostra che queste feature contribuiscono poco. Potrebbero essere rimosse in versioni future.

### 3.3 MITRE ATT&CK Techniques

**Approccio:**
1. Parsing della stringa separata da `;`
2. Selezione Top 20 tecniche per frequenza (calcolata su train)
3. One-hot encoding delle top tecniche
4. Conteggio totale tecniche per incidente

**Perché Top 20?** Bilanciamento tra:
- Catturare le tecniche più rilevanti
- Evitare esplosione dimensionale (>400 tecniche uniche nel dataset)
- Mantenere feature con sufficiente supporto statistico

### 3.4 Aggregazione Evidence → Incident

| Feature | Aggregazione | Rationale |
|---------|--------------|-----------|
| Target | max | Se almeno una evidence è TP, l'incidente è TP |
| AlertId | nunique | Numero di alert distinti |
| Id | count | Numero totale evidenze |
| AlertRisk | mean, max | Rischio medio e massimo degli alert |
| Hour | mean, std | Distribuzione temporale delle evidenze |
| IsWeekend | mean | % evidenze nel weekend |
| Mitre_* | max | Presenza della tecnica nell'incidente |

---

## 4. Gestione dello Sbilanciamento

### Training Set: Undersampling

**Scelta:** Random undersampling della classe maggioritaria

**Alternativa considerata:** SMOTE (oversampling)

**Perché undersampling:**
- Dataset molto grande (>1M incidenti) - non abbiamo bisogno di dati sintetici
- Computazionalmente più efficiente
- Evita il rischio di overfitting su campioni sintetici
- Informazione realistica

---

## 5. Selezione e Tuning dei Modelli

### Modelli Confrontati

| Modello | Rationale |
|---------|-----------|
| Decision Tree | Baseline interpretabile, identifica feature splitting points |
| Random Forest | Ensemble robusto, riduce varianza del DT |
| XGBoost | SOTA su dati tabulari, gestisce bene sbilanciamento |
| MLP (PyTorch) | Verifica se relazioni non-lineari complesse migliorano le performance |

### Hyperparameter Tuning

**Metodo:** RandomizedSearchCV con 5-fold stratified cross-validation

**Perché RandomizedSearchCV invece di GridSearchCV?**
- Spazio degli iperparametri ampio
- RandomizedSearchCV esplora più efficientemente lo spazio
- Consente di fissare un budget computazionale (n_iter)

**Parametri ottimizzati per XGBoost:**
```python
param_distributions = {
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}
```

### Early Stopping per MLP

Il training loop PyTorch implementa:
1. Monitoraggio della validation loss ad ogni epoca
2. Salvataggio del best model (lowest val_loss)
3. Early stopping con patience=10 (stop se val_loss non migliora per 10 epoche)

---

## 6. Metriche di Valutazione

### Metrica Primaria: F1-Score

**Perché F1?**
- Bilancia Precision e Recall
- Più informativo dell'accuracy su dataset  (anche se in questo caso l'effetto è stato mitigato)
- Rilevante per il dominio: vogliamo sia alta detection (recall) che pochi falsi allarmi (precision)

### Metriche Secondarie

| Metrica | Uso |
|---------|-----|
| AUC-ROC | Valutazione indipendente dalla soglia |
| Precision | Quanto sono affidabili i positive predictions |
| Recall | Quante minacce reali catturiamo |
| Confusion Matrix | Analisi errori per classe |

---

## 7. Riproducibilità

Tutti gli esperimenti usano `random_state=42` per:
- Split dei dati
- Inizializzazione modelli
- Sampling per undersampling

**File di output:**
- `data/processed_final/feature_mappings.pkl`: Mappe di encoding per inferenza
- `models/*/model.json` o `model.pth`: Pesi dei modelli salvati
- `models/*/metrics.json`: Metriche finali

---

## Riferimenti

- Dataset GUIDE: [Microsoft Security AI Research](https://github.com/microsoft/GUIDE)
- MITRE ATT&CK Framework: https://attack.mitre.org/
- XGBoost: Chen & Guestrin (2016), "XGBoost: A Scalable Tree Boosting System"
