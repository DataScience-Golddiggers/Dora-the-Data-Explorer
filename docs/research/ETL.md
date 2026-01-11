### 1. Feature Engineering di Base (`Test_03 - FeatureEngineering_v3.ipynb`)

Questo notebook è il punto di partenza e trasforma il dataset grezzo (`GUIDE_Train.csv`) in un formato aggregato a livello di `IncidentId`, pronto per il modeling.

1.  **Target Binario**: La colonna `IncidentGrade` (multi-classe) viene trasformata in `BinaryIncidentGrade` (1 per `TruePositive`, 0 per `FalsePositive`/`BenignPositive`).
2.  **SmoothedRisk**: Viene calcolato un punteggio di rischio per ogni `AlertTitle` utilizzando il *Bayesian smoothing*. Questo corregge le stime di rischio per gli alert con poche occorrenze, evitando overfitting.
3.  **Encoding Geografico**: Le colonne `CountryCode`, `State` e `City` vengono combinate e sostituite da una singola feature di frequenza (`GeoLoc_freq`).
4.  **Feature Temporali**: Dal `Timestamp` vengono estratte informazioni come mese, ora, giorno della settimana e un flag per il weekend.
5.  **Encoding Categoriale**:
    *   **Frequency Encoding**: Colonne ad alta cardinalità (es. `ThreatFamily`, `ResourceType`) vengono codificate con la loro frequenza normalizzata.
    *   **One-Hot Encoding**: Colonne a bassa cardinalità e con segnale forte (es. `SuspicionLevel`, `EvidenceRole`) vengono trasformate in colonne binarie.
6.  **MITRE Techniques**: Le tecniche MITRE vengono estratte, si identificano le 30 più comuni e si crea una matrice one-hot-encoded.
7.  **Aggregazione**: Tutti i dati a livello di "evidence" vengono aggregati a livello di `IncidentId` usando operazioni come `mean`, `sum`, `nunique`, `min`, `max` e `mode`. Viene anche calcolata la durata dell'incidente (`Duration_seconds`).
8.  **Split e Salvataggio**: Il dataset aggregato viene diviso in set di training e test (70/30) con stratificazione e salvato nella cartella `data/processed_v3/`.

### 2. Ribilanciamento con Dati Esterni (`Test_11 - DataRebalancing.ipynb`)

Questo notebook tenta di bilanciare il training set utilizzando un metodo che introduce **un grave rischio di data leakage**.

1.  **Caricamento Dati**: Carica il training set sbilanciato (`processed_v3`) e il file `GUIDE_Test.csv`.
2.  **Preprocessing del Test Set**: Applica **lo stesso identico pipeline di feature engineering** descritto sopra al file `GUIDE_Test.csv`.
3.  **Estrazione e Merge**:
    *   Filtra i campioni `TruePositive` (classe minoritaria) dal set di test appena processato.
    *   **Aggiunge questi campioni al training set originale**.
4.  **Nuovo Split**: Il nuovo dataset "bilanciato" (composto da train originale + parte del test) viene nuovamente splittato in train/test e salvato in `data/processed_v3_balanced/`.

**Criticità**: Questo approccio "contamina" il training set con informazioni provenienti dal test set, portando a una valutazione delle performance del modello eccessivamente ottimistica e non realistica (data leakage).

### 3. Ribilanciamento Ibrido Corretto (`Test_14_DataRebalancing_Hybrid.ipynb`)

Questo notebook implementa una strategia di ribilanciamento standard e corretta, che opera **esclusivamente sul training set** per evitare data leakage.

1.  **Caricamento Dati**: Carica i set di training e test dalla cartella `processed_v3_balanced` (o una versione precedente non contaminata).
2.  **Step 1: RandomUnderSampler**: Riduce il numero di campioni della classe maggioritaria (classe 0, "Non-TP") per portarla a un rapporto definito (es. 1.5 volte la classe minoritaria).
3.  **Step 2: SMOTE (Oversampling)**: Aumenta il numero di campioni della classe minoritaria (classe 1, "TP") generando campioni sintetici fino a raggiungere un bilanciamento perfetto (rapporto 1:1).
4.  **Salvataggio**: Il nuovo training set bilanciato (`X_train_balanced`, `y_train_balanced`) viene salvato in una nuova cartella (`processed_v4_hybrid/`). **Il test set originale viene copiato senza alcuna modifica**, garantendo una valutazione onesta del modello.
