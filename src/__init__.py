"""
Msingi1 language model package.
Contains model architecture, data processing, and training utilities.
"""

from .data_processor import SwahiliDataset, extract_dataset, get_dataset_stats
from .model import Msingi1, MsingiConfig

__all__ = [
    'SwahiliDataset',
    'Msingi1',
    'MsingiConfig',
    'extract_dataset',
    'get_dataset_stats'
]
