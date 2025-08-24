# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GNU GPLv3

import logging
import os
from pathlib import Path

from pmarlo import MarkovStateModel, Pipeline, Protein, ReplicaExchange, Simulation
from pmarlo.replica_exchange.config import RemdConfig

# Configure logging to show all messages in console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)

# Use relative path from the project root
protein_path = os.path.join("../tests", "data", "3gd8-fixed.pdb")


def verify_components():
    """Verify that all PMARLO components can be initialized."""
    print("\n=== Verifying PMARLO Components ===")
    try:
        protein = Protein(
            protein_path, ph=7.0, auto_prepare=False
        )  # Using pre-fixed PDB
        print(" Protein component initialized")

        replica_exchange = ReplicaExchange.from_config(
            RemdConfig(
                pdb_file=protein_path, temperatures=[300, 310, 320], auto_setup=False
            )
        )
        # Plan stride minimally for short verification
        replica_exchange.plan_reporter_stride(
            total_steps=500, equilibration_steps=50, target_frames=100
        )
        replica_exchange.setup_replicas()
        print(" Replica Exchange component initialized")

        simulation = Simulation(protein_path, temperature=300, steps=1000)
        print(" Simulation component initialized")

        markov_state_model = MarkovStateModel()
        print(" Markov State Model component initialized")

        print("\nAll components initialized successfully!")
        return True
    except Exception as e:
        print(f"\nError during component initialization: {e}")
        return False


def run_pipeline():
    """Run the complete PMARLO pipeline."""
    print("\n=== Running PMARLO Pipeline ===")
    try:
        pipeline = Pipeline(
            protein_path, temperatures=[300, 310, 320], steps=1000, auto_continue=False
        )

        print("Starting pipeline execution...")
        results = pipeline.run()

        print("\nPipeline Results:")
        print("-----------------")
        for key, value in results.items():
            if isinstance(value, dict) and "status" in value:
                print(f"• {key}: {value.get('status', 'unknown')}")
            elif isinstance(value, dict):
                print(f"• {key}: {len(value)} items")
        return True
    except Exception as e:
        print(f"\nError during pipeline execution: {e}")
        return False


if __name__ == "__main__":
    print("Starting PMARLO Verification\n" + "=" * 30)

    components_ok = verify_components()
    pipeline_ok = run_pipeline()

    print("\n" + "=" * 30)
    if components_ok and pipeline_ok:
        print(" PMARLO verification completed successfully!")
    else:
        print(" Some verifications failed. Check the logs above.")
