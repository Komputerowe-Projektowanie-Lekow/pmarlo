# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Bias variable setup utilities for metadynamics simulations.
"""

import logging
from typing import List

import mdtraj as md
import numpy as np
from openmm import CustomTorsionForce
from openmm.app.metadynamics import BiasVariable

logger = logging.getLogger("pmarlo")


def setup_bias_variables(pdb_file: str) -> List:
    """
    Set up bias variables for metadynamics.

    Args:
        pdb_file: Path to the PDB file

    Returns:
        List of bias variables
    """
    # Load trajectory to get dihedral indices
    traj0 = md.load_pdb(pdb_file)
    phi_indices, _ = md.compute_phi(traj0)

    if len(phi_indices) == 0:
        logger.warning("No phi dihedrals found - proceeding without bias variables")
        return []

    bias_variables = []

    # Add phi dihedral as bias variable
    for i, phi_atoms in enumerate(phi_indices[:2]):  # Use first 2 phi dihedrals
        phi_atoms = [int(atom) for atom in phi_atoms]

        phi_force = CustomTorsionForce("theta")
        phi_force.addTorsion(*phi_atoms, [])

        phi_cv = BiasVariable(
            phi_force,
            -np.pi,  # minValue
            np.pi,  # maxValue
            0.35,  # biasWidth (~20 degrees)
            True,  # periodic
        )

        bias_variables.append(phi_cv)
        logger.info(f"Added phi dihedral {i+1} as bias variable: atoms {phi_atoms}")

    return bias_variables
