# Changelog - Revisione Critica Notebook

**Data:** Gennaio 2026  
**Autore:** Revisione AI

---

## Problematiche Identificate e Risolte

### 1. ❌ Validation Set Bilanciato → ✅ Distribuzione Reale

**Problema:** Il validation set veniva sottoposto a undersampling, distorcendo la valutazione delle performance reali.

**Soluzione:** 
- Modificato [15-Model_Training_and_Comparison.ipynb](../notebook/15-Model_Training_and_Comparison.ipynb) per usare `val_final_imbalanced.csv`
- La distribuzione reale permette di stimare le performance vere del modello

---

### 2. ❌ Iperparametri Arbitrari → ✅ Cross-Validation

**Problema:** Gli iperparametri erano scelti senza giustificazione (es. `max_depth=10`, `n_estimators=100`).

**Soluzione:** 
- Aggiunta sezione "Hyperparameter Tuning con Cross-Validation" nel notebook 15
- Implementato `RandomizedSearchCV` con 5-fold stratified CV per:
  - Decision Tree
  - Random Forest
  - XGBoost
- I best_estimator vengono riutilizzati per la valutazione finale

---

### 3. ❌ MLP senza Early Stopping → ✅ Training Loop Completo

**Problema:** Il training PyTorch non monitorava la validation loss, rischiando overfitting.

**Soluzione:**
- Implementato training loop con:
  - Validation monitoring ad ogni epoca
  - Early stopping con patience=10
  - Learning rate scheduling (ReduceLROnPlateau)
  - Salvataggio del best model
- Aggiunta visualizzazione delle learning curves

---

### 4. ❌ Mancanza di Documentazione → ✅ docs/methodology.md

**Problema:** Le scelte progettuali non erano giustificate.

**Soluzione:**
- Creato [docs/methodology.md](methodology.md) con:
  - Giustificazione classificazione binaria vs ternaria
  - Strategia anti-leakage
  - Spiegazione delle tecniche di feature engineering
  - Rationale per ogni modello
  - Metriche di valutazione

---

### 5. ❌ Nessuna Analisi Multicollinearità → ✅ Sezione Aggiunta

**Problema:** Non veniva verificata la ridondanza tra feature correlate.

**Soluzione:**
- Aggiunta sezione 5.1 in [14-FeatureEngineering_Pipeline.ipynb](../notebook/14-FeatureEngineering_Pipeline.ipynb)
- Identifica coppie con |r| > 0.9
- Heatmap delle correlazioni per le top feature

---

## File Modificati

| File | Modifiche |
|------|-----------|
| `notebook/15-Model_Training_and_Comparison.ipynb` | CV tuning, validation corretto, MLP migliorato |
| `notebook/14-FeatureEngineering_Pipeline.ipynb` | Analisi multicollinearità |
| `docs/methodology.md` | **Nuovo** - Documentazione metodologica |
| `docs/CHANGELOG.md` | **Nuovo** - Questo file |

---

## Struttura Notebook 15 Aggiornata

1. Caricamento e Preparazione Dati
2. Hyperparameter Tuning con Cross-Validation *(nuovo)*
3. Addestramento Modelli con Parametri Ottimali
4. Valutazione Finale su GUIDE_Test
5. Visualizzazioni e Confronto
6. Feature Importance (XGBoost)
7. MLP con PyTorch (con Early Stopping) *(migliorato)*
8. Salvataggio Modelli e Risultati *(nuovo)*

---

## Output Generati

Dopo l'esecuzione del notebook 15, vengono salvati:

```
models/
├── cv_results.json          # Iperparametri ottimali + CV scores
├── decisiontree_model.pkl   # Modello Decision Tree
├── randomforest_model.pkl   # Modello Random Forest  
├── xgboost_model.pkl        # Modello XGBoost
├── mlp_model.pkl            # Modello MLP Sklearn
├── mlp_pytorch_model.pth    # Modello MLP PyTorch
└── final_results.json       # Metriche finali complete
```

---

## Miglioramenti Futuri Suggeriti

1. **Ablation Study**: Testare l'impatto della rimozione di gruppi di feature
2. **Calibration Plots**: Verificare se le probabilità predette sono calibrate
3. **Learning Curves**: Analizzare se servono più dati
4. **Classificazione Ternaria**: Aggiungere sezione opzionale con 3 classi
