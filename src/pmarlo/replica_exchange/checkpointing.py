# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Checkpoint state management for replica exchange simulations.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import openmm

logger = logging.getLogger("pmarlo")


class CheckpointManager:
    """Manages checkpoint state for replica exchange simulations."""

    @staticmethod
    def save_checkpoint_state(
        contexts: List[openmm.Context],
        n_replicas: int,
        temperatures: List[float],
        replica_states: List[int],
        state_replicas: List[int],
        exchange_attempts: int,
        exchanges_accepted: int,
        exchange_history: List[List[int]],
        output_dir: str,
        exchange_frequency: int,
        random_seed: int,
        rng_state: Dict,
        reporter_stride: Optional[int],
        replica_reporter_strides: List[int],
    ) -> Dict[str, Any]:
        """
        Save the current state for checkpointing.

        Returns:
            Dictionary containing the current state
        """
        if not contexts or len(contexts) != n_replicas:
            return {"setup": False}

        # Save critical state information
        state = {
            "setup": True,
            "n_replicas": n_replicas,
            "temperatures": temperatures,
            "replica_states": replica_states.copy(),
            "state_replicas": state_replicas.copy(),
            "exchange_attempts": exchange_attempts,
            "exchanges_accepted": exchanges_accepted,
            "exchange_history": exchange_history.copy(),
            "output_dir": str(output_dir),
            "exchange_frequency": exchange_frequency,
            "random_seed": random_seed,
            "rng_state": rng_state,
        }

        # Save states in XML for long-term stability across versions
        from openmm import XmlSerializer

        replica_xml_states: List[str] = []
        for i, context in enumerate(contexts):
            try:
                sim_state = context.getState(
                    getPositions=True, getVelocities=True, getEnergy=True
                )
                xml_str = XmlSerializer.serialize(sim_state)
                replica_xml_states.append(xml_str)
            except Exception as e:
                logger.warning(f"Could not save state XML for replica {i}: {e}")
                replica_xml_states.append("")

        state["replica_state_xml"] = replica_xml_states
        # Persist reporter stride data for demux after resume
        state["reporter_stride"] = int(reporter_stride or 1)
        state["replica_reporter_strides"] = replica_reporter_strides.copy()
        return state

    @staticmethod
    def restore_from_checkpoint(
        checkpoint_state: Dict[str, Any],
        contexts: List[openmm.Context],
        n_replicas: int,
    ) -> Dict[str, Any]:
        """
        Extract restoration data from checkpoint state.

        Args:
            checkpoint_state: Previously saved state dictionary
            contexts: List of OpenMM contexts to restore
            n_replicas: Number of replicas

        Returns:
            Dictionary with restored state values
        """
        if not checkpoint_state.get("setup", False):
            logger.info(
                "Checkpoint indicates replicas were not set up, will need setup..."
            )
            return {"needs_setup": True}

        logger.info("Restoring replica exchange from checkpoint...")

        restored = {
            "needs_setup": False,
            "exchange_attempts": checkpoint_state.get("exchange_attempts", 0),
            "exchanges_accepted": checkpoint_state.get("exchanges_accepted", 0),
            "exchange_history": checkpoint_state.get("exchange_history", []),
            "replica_states": checkpoint_state.get(
                "replica_states", list(range(n_replicas))
            ),
            "state_replicas": checkpoint_state.get(
                "state_replicas", list(range(n_replicas))
            ),
            "random_seed": checkpoint_state.get("random_seed", None),
            "rng_state": checkpoint_state.get("rng_state"),
            "reporter_stride": checkpoint_state.get("reporter_stride"),
            "replica_reporter_strides": checkpoint_state.get(
                "replica_reporter_strides"
            ),
        }

        # Restore replica states from XML if available
        from openmm import XmlSerializer

        replica_xml = checkpoint_state.get("replica_state_xml", [])
        if replica_xml and len(replica_xml) == n_replicas and contexts:
            logger.info("Restoring individual replica states from XML...")
            for i, (context, xml_str) in enumerate(zip(contexts, replica_xml)):
                if xml_str:
                    try:
                        state_obj = XmlSerializer.deserialize(xml_str)
                        if state_obj.getPositions() is not None:
                            context.setPositions(state_obj.getPositions())
                        if state_obj.getVelocities() is not None:
                            context.setVelocities(state_obj.getVelocities())
                        logger.info(f"Restored state for replica {i}")
                    except Exception as e:
                        logger.warning(f"Could not restore state for replica {i}: {e}")

        logger.info(
            "Checkpoint restoration complete. Exchange stats: "
            f"{restored['exchanges_accepted']}/{restored['exchange_attempts']}"
        )

        return restored
