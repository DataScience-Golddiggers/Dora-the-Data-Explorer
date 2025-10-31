"""
GUIDE Dataset Pipeline - Dora the Data Explorer

Pipeline completa per classificazione di incidenti di cybersecurity.
"""

__version__ = "1.0.0"
__author__ = "DataScience-Golddiggers"

from . import preprocessing
from . import training
from . import inference

__all__ = ['preprocessing', 'training', 'inference']
