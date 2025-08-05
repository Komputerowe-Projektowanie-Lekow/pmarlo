"""
PMARLO: Protein Markov State Model Analysis with Replica Exchange

A Python package for protein simulation and Markov state model chain generation,
providing an OpenMM-like interface for molecular dynamics simulations.
"""

from .protein.protein import Protein
from .replica_exchange.replica_exchange import ReplicaExchange
from .markov_state_model.markov_state_model import EnhancedMSM as MarkovStateModel
from .simulation.simulation import Simulation
from .pipeline import Pipeline, LegacyPipeline

__version__ = "0.1.0"
__author__ = "PMARLO Development Team"

# Main classes for the clean API
__all__ = [
    'Protein',
    'ReplicaExchange', 
    'MarkovStateModel',
    'Simulation',
    'Pipeline',
    'LegacyPipeline'
]