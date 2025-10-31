"""
Configurazione globale per la pipeline.
"""

import os

# Percorsi base
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# File dati
TRAIN_FILE = os.path.join(DATA_DIR, 'GUIDE_Train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'GUIDE_Test.csv')

# Directory dati processati
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed_v3')
PROCESSED_BALANCED_DIR = os.path.join(DATA_DIR, 'processed_v3_balanced')

# Pipeline
PIPELINE_PATH = os.path.join(MODELS_DIR, 'preprocessing_pipeline.pkl')

# Hyperparameters default
XGBOOST_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 15,
    'min_samples_split': 50,
    'min_samples_leaf': 20
}

MLP_PARAMS = {
    'hidden_dims': [128, 64, 32],
    'dropout': 0.3,
    'learning_rate': 0.001,
    'batch_size': 256,
    'epochs': 50,
    'early_stopping_patience': 10
}

# Feature Engineering
FEATURE_ENGINEERING_PARAMS = {
    'alpha': 2.0,
    'beta': 2.0,
    'top_n_mitre': 30,
    'rare_verdict_threshold': 100,
    'rare_category_threshold': 100
}

# Data Rebalancing
REBALANCING_PARAMS = {
    'test_size': 0.3,
    'random_state': 42
}

# Seed globale
RANDOM_STATE = 42
