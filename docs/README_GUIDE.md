# Notebook EDA per Dataset GUIDE - Microsoft Cybersecurity Incidents

## Descrizione Dataset

Il **GUIDE (Guided Response Dataset)** è il più grande dataset pubblico di incidenti di cybersecurity reali fornito da Microsoft.

### Caratteristiche Principali:
- **13+ milioni** di evidenze
- **33 tipi** di entità
- **1.6 milioni** di alert
- **1 milione** di incidenti annotati con triage labels
- **6,100+** organizzazioni
- **9,100** DetectorId unici
- **441** tecniche MITRE ATT&CK
- **Periodo**: 2 settimane (giugno 2024)

## Struttura del Notebook

Il notebook `EDA-Cybersec.ipynb` è stato adattato per analizzare il dataset GUIDE e include le seguenti sezioni:

### 1. Setup e Importazione Librerie
- Pandas, NumPy, Matplotlib, Seaborn

### 2. Caricamento Dati
- Caricamento del file `GUIDE_Train.csv` (~9.5M righe, 45 colonne)

### 3. Ispezione Iniziale
- Visualizzazione dimensioni, prime/ultime righe
- Info sui tipi di dati
- Statistiche descrittive

### 4. Pulizia Dati
- **Gestione Missing Values** con strategie specifiche per:
  - `ActionGrouped`/`ActionGranular` (~99% missing)
  - `MitreTechniques` (~57% missing)
  - `SuspicionLevel` (~85% missing)
  - `LastVerdict` (~77% missing)
  - `IncidentGrade` - rimossi record mancanti (target)
- **Rimozione Duplicati** basata su `Id` univoco

### 5. Analisi Univariata
- **Colonne Numeriche**: Analisi cardinalità (ID anonimizzati)
- **Colonne Categoriche**: Distribuzione per:
  - `IncidentGrade` (TP, BP, FP)
  - `Category` (InitialAccess, Exfiltration, etc.)
  - `EntityType` (33 tipi)
  - `EvidenceRole`
  - `SuspicionLevel`
  - `LastVerdict`

### 6. Analisi Bivariata
- Relazioni tra `Category` e `IncidentGrade`
- Relazioni tra `EntityType` e `IncidentGrade`
- Relazioni tra `EvidenceRole` e `IncidentGrade`
- Pattern temporali (ora del giorno, giorno settimana)
- Co-occorrenze: Alert per Incident, Evidenze per Alert

### 7. Analisi Specifica per Cybersecurity
- Distribuzione `IncidentGrade` (target)
- Top categorie di minacce MITRE ATT&CK
- Pattern temporali (orari e giornalieri)
- Analisi MITRE Techniques più comuni
- Distribuzione per organizzazione
- Analisi geografica (Country, State, City)
- Analisi OS Family e OS Version
- Analisi Remediation Actions (subset con ActionGrouped/ActionGranular)

### 8. Conclusioni e Prossimi Passi
- Riepilogo qualità dati
- Pattern identificati
- Strategie di Feature Engineering
- Approcci di modeling suggeriti

## Colonne del Dataset (45 totali)

### Identificatori
- `Id` - ID univoco record
- `OrgId` - ID organizzazione (anonimizzato)
- `IncidentId` - ID incidente
- `AlertId` - ID alert
- `DetectorId` - ID rilevatore

### Temporali
- `Timestamp` - Data/ora evento

### Target e Metadati
- `IncidentGrade` - **VARIABILE TARGET** (TruePositive, BenignPositive, FalsePositive)
- `AlertTitle` - Titolo alert (ID anonimizzato)
- `Category` - Categoria minaccia MITRE
- `MitreTechniques` - Tecniche MITRE ATT&CK

### Azioni di Remediation (opzionale, ~1% dei dati)
- `ActionGrouped` - Azione raggruppata
- `ActionGranular` - Azione granulare

### Entità e Ruoli
- `EntityType` - Tipo di entità (33 tipi)
- `EvidenceRole` - Ruolo dell'evidenza

### Dettagli Tecnici (ID anonimizzati)
- `DeviceId`, `DeviceName`
- `Sha256` - Hash file
- `IpAddress`
- `Url`
- `AccountSid`, `AccountUpn`, `AccountObjectId`, `AccountName`
- `NetworkMessageId`
- `EmailClusterId`
- `RegistryKey`, `RegistryValueName`, `RegistryValueData`
- `ApplicationId`, `ApplicationName`, `OAuthApplicationId`
- `ThreatFamily`
- `FileName`, `FolderPath`
- `ResourceIdName`, `ResourceType`
- `Roles`

### Sistema Operativo
- `OSFamily` - Famiglia OS
- `OSVersion` - Versione OS

### Email e Verdict
- `AntispamDirection`
- `SuspicionLevel` - Livello sospetto
- `LastVerdict` - Ultimo verdetto

### Geografia
- `CountryCode` - Codice paese
- `State` - Stato/Regione
- `City` - Città

## Obiettivi di Machine Learning

### Task Principale: Triage Prediction
**Predire `IncidentGrade` (3 classi)**
- TruePositive (TP): ~35%
- BenignPositive (BP): ~43%
- FalsePositive (FP): ~21%

**Metrica Raccomandata**: Macro-F1 Score

### Task Secondario: Remediation Action Prediction
**Predire `ActionGrouped` o `ActionGranular`**
- Solo ~1% dei dati ha queste labels
- Subset di 26k incidenti con azioni

## Prossimi Passi Suggeriti

### Feature Engineering
1. **Features Temporali**
   - Ora, giorno settimana, periodo giornata
   - Time since previous incident (per org)

2. **Aggregazioni a livello Incident**
   - Numero di alert per incident
   - Diversità di EntityType
   - Numero di evidenze per alert

3. **Features Categoriche**
   - One-Hot Encoding per Category, EntityType
   - MITRE Techniques parsing e conteggio
   - Target encoding per DetectorId, OrgId

4. **Features da Missing Values**
   - Indicatori binari per presenza/assenza features opzionali
   - Conteggio campi popolati per record

### Modeling
1. **Baseline Models**
   - Logistic Regression
   - Random Forest
   - Gradient Boosting (XGBoost/LightGBM)

2. **Gestione Sbilanciamento**
   - SMOTE per oversampling
   - Class weights
   - Stratified K-Fold CV

3. **Advanced Models**
   - CatBoost (gestisce categorie native)
   - Neural Networks
   - Ensemble methods

### Validazione
- Stratified split mantenendo distribuzione IncidentGrade
- Cross-validation con focus su macro-F1
- Analisi per-class (precision, recall per TP, BP, FP)

## Come Eseguire il Notebook

1. Assicurati di avere il file `GUIDE_Train.csv` nella cartella `../data/`
2. Installa le dipendenze:
   ```bash
   pip install pandas numpy matplotlib seaborn
   ```
3. Apri `EDA-Cybersec.ipynb` in Jupyter Notebook o JupyterLab
4. Esegui le celle in sequenza

**Nota**: Il caricamento del dataset richiede alcuni secondi data la dimensione (~9.5M righe).

## Riferimenti

- **Paper**: [GUIDE: Guided Response Dataset on arXiv](https://arxiv.org/)
- **MITRE ATT&CK**: [Framework delle tecniche di attacco](https://attack.mitre.org/)
- **Microsoft Security AI Research**: Dataset host

---
*Ultimo aggiornamento: Ottobre 2025*

