# Dizionario delle Feature - GUIDE Dataset
Questo documento descrive le feature generate nel notebook `2-FeatureEngineering.ipynb` e utilizzate per l'addestramento dei modelli di classificazione degli incidenti.

Tutte le feature sono aggregate a livello di **IncidentId**.

## 1. Target Variables
| Feature | Tipo | Descrizione |
|---------|------|-------------|
| **BinaryIncidentGrade** | Binario (0/1) | Il target per la classificazione. <br>• `1`: **TruePositive** (Incidente reale)<br>• `0`: **BenignPositive** (Falso allarme legittimo) o **FalsePositive** (Errore di rilevamento). |
| **IncidentGrade** | Categorico | La label originale (usata per analisi, non per training diretto in questo setup binario). |

## 2. Feature Strutturali (Volumetria)
Queste feature descrivono la "dimensione" e la complessità dell'incidente.

| Feature | Logica Aggregazione | Significato |
|---------|---------------------|-------------|
| **NumAlerts** | `nunique` su AlertId | Quanti alert distinti compongono questo incidente. Un numero alto indica un attacco complesso o rumoroso. |
| **NumEvidences** | `count` su Id | Il numero totale di righe (evidenze) associate all'incidente. |

## 3. Risk & Security Indicators
Indicatori derivati dalla gravità degli alert e delle evidenze.

| Feature | Logica Aggregazione | Significato |
|---------|---------------------|-------------|
| **SmoothedRisk_avg** | `mean` | La media del "rischio smussato" (Bayesian Smoothing) calcolato su `AlertTitle`. <br>• Valori alti: Gli alert in questo incidente sono storicamente associati a TruePositive.<br>• Valori bassi: Gli alert sono spesso falsi positivi. |
| **SuspicionLevel_{Level}** | `sum` | Conteggio di quante evidenze hanno un certo livello di sospetto (es. `SuspicionLevel_High`, `SuspicionLevel_Medium`). Generate via One-Hot Encoding e sommate. |

## 4. Feature Temporali
Analisi del comportamento temporale dell'attacco.

| Feature | Logica Aggregazione | Significato |
|---------|---------------------|-------------|
| **Duration_seconds** | `max(ts) - min(ts)` | Durata totale dell'incidente in secondi. Attacchi automatici possono durare millisecondi, APT giorni. |
| **Hour_First** | `min` | L'ora del giorno (0-23) in cui è apparso il primo segnale. |
| **Hour_Last** | `max` | L'ora del giorno dell'ultimo segnale. |
| **Hour_Avg** | `mean` | L'ora media ponderata delle evidenze. |
| **month** | `mode` | Il mese prevalente dell'incidente (stagionalità). |
| **weekday** | `mode` | Il giorno della settimana prevalente (1=Lun, 7=Dom). |
| **IsWeekend** | `max` | Flag binario (1 se almeno un'evidenza è avvenuta nel weekend). |

## 5. Feature Geografiche
| Feature | Logica Aggregazione | Significato |
|---------|---------------------|-------------|
| **GeoLoc_freq_avg** | `mean` | Media della frequenza della combinazione `Country-State-City` delle evidenze.<br>• Valori **bassi**: L'incidente avviene in luoghi rari/anomali per il dataset.<br>• Valori **alti**: Luoghi standard/comuni. |

## 6. Categorical Frequency Features
Per evitare l'alta dimensionalità, queste colonne categoriche sono state trasformate usando la loro **frequenza relativa** nel dataset.
*Valore aggregato: Media (mean) delle frequenze delle evidenze nell'incidente.*

| Feature | Descrizione Originale | Interpretazione del Valore |
|---------|-----------------------|----------------------------|
| **ThreatFamily_freq** | Famiglia del malware/threat | Basso = Malware raro/nuovo (spesso più pericoloso). Alto = Minaccia comune. |
| **AntispamDirection_freq** | Direzione traffico (Inbound/Outbound) | Frequenza della direzione del traffico. |
| **ActionGranular_freq** | Azione specifica intrapresa | Quanto è comune l'azione intrapresa dal sistema (es. "Block", "Allow"). |
| **LastVerdict_freq** | Giudizio finale del sistema | Quanto è comune il verdetto automatico assegnato. |
| **ResourceType_freq** | Tipo risorsa (File, Process, Url) | Quanto è comune il tipo di risorsa colpita. |
| **Roles_freq** | Ruolo entità (Attacker, Victim) | Rarità della combinazione di ruoli. |
| **ActionGrouped_freq** | Azione macro (Block, Monitor) | Rarità della macro-azione. |
| **EntityType_freq** | Tipo entità (Host, IP, Account) | Se l'incidente coinvolge entità di tipo raro. |
| **Category_freq** | Categoria alert | Se l'incidente appartiene a categorie di alert rare. |

## 7. Evidence Roles (Ruoli Contestuali)
Generate via One-Hot Encoding e sommate (`sum`). Indicano la struttura dell'incidente.

| Feature | Significato |
|---------|-------------|
| **EvidenceRole_{Role}** | Conta quante evidenze hanno uno specifico ruolo (es. `EvidenceRole_Attacker`, `EvidenceRole_Target`). Aiuta a distinguere incidenti con molti attaccanti o molte vittime. |

## 8. MITRE ATT&CK Techniques
Le 30 tecniche MITRE più frequenti nel dataset.
*Logica: One-Hot Encoded e poi Sommati (`sum`) per incidente.*

| Feature | Esempio | Significato |
|---------|---------|-------------|
| **Txxxx** | T1059 (Command and Scripting) | Conta quante volte questa specifica tecnica è stata osservata nell'incidente. Indica la presenza di TTPs (Tactics, Techniques, and Procedures) specifici. |

---
**Nota sulla Dimensionalità:**
Il dataset finale contiene circa **70-80 colonne**. Dato che abbiamo ~450.000 incidenti nel training set, il rapporto campioni/feature è > 5000:1, escludendo rischi di Curse of Dimensionality. Le feature "Frequency" comprimono l'informazione di migliaia di possibili categorie in singoli valori continui rappresentativi della loro "normalità".
