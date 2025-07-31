
import numpy as np, mdtraj as md
import openmm.unit
from openmm.app import *
from openmm import *
from openmm.unit import *
from openmm.app import Modeller
from openmm.app.metadynamics import Metadynamics
from openmm import Platform
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict
from pdbfixer import PDBFixer
import os
import matplotlib.pyplot as plt
from pathlib import Path


# pdb = PDBFile(str(Path(__file__).resolve().parent.parent / "tests" / "3gd8-fixed.pdb"))
# traj0 = md.load_pdb(str(Path(__file__).resolve().parent.parent / "tests" / "3gd8-fixed.pdb"))

BASE_DIR = Path(__file__).resolve().parent.parent
TESTS_DIR = BASE_DIR / "tests"

BASE_DIR = Path(__file__).resolve().parent.parent
dcd_path = BASE_DIR / "tests" / "traj.dcd"
pdb_path = BASE_DIR / "tests" / "3gd8-fixed.pdb"

print("DCD file exists:", os.path.exists(dcd_path))
print("PDB file exists:", os.path.exists(pdb_path))

def prepare_system(pdb_file_name):
    #protein = fix_protein(pdb_file_name)
    pdb = PDBFile(pdb_file_name)
    forcefield = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")

    system = forcefield.createSystem(pdb.topology,
                                     nonbondedMethod=PME,
                                     constraints=HBonds)


    traj0 = md.load_pdb(str(TESTS_DIR / "3gd8-fixed.pdb"))
    phi_indices, _ = md.compute_phi(traj0)
    if len(phi_indices) == 0:
        raise RuntimeError("No φ dihedral found in the PDB structure – cannot set up CV.")

    phi_atoms = [int(i) for i in phi_indices[0]]

    phi_force = CustomTorsionForce("theta")
    phi_force.addTorsion(*phi_atoms, [])

    phi_cv = BiasVariable(
        phi_force,
        minValue=-np.pi,
        maxValue=np.pi,
        biasWidth=0.35,  # ~20°
        periodic=True,
    )

    bias_dir = BASE_DIR / "bias"
    
    # Clear existing bias files to avoid conflicts
    if bias_dir.exists():
        for file in bias_dir.glob("bias_*.npy"):
            try:
                file.unlink()
            except Exception:
                pass
    
    os.makedirs(str(bias_dir), exist_ok=True)

    meta = Metadynamics(
        system,
        [phi_cv],
        temperature=300 * kelvin,
        biasFactor=10.0,
        height=1.0 * kilojoules_per_mole,
        frequency=500,  # hill every 1 ps (500 × 2 fs)
        biasDir=str(bias_dir),
        saveFrequency=1000,
    )


    integrator = LangevinIntegrator(300*kelvin, # T
                                    1/picosecond,   # γ
                                    2*femtoseconds) # Δt

    # DO *NOT* add phi_force to the System – Metadynamics will own it.
    platform = Platform.getPlatformByName("CPU")  # or "CPU", "OpenCL", etc.

    simulation = Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)

    simulation.minimizeEnergy(maxIterations=100)
    simulation.step(1000)
    print("\u2714 Build & equilibration complete\n")
    return simulation, meta

def production_run(steps, simulation, meta):
    print("Stage 3/5  –  production run (test: short simulation)...")
    dcd_filename = str(TESTS_DIR / "traj.dcd")
    dcd = DCDReporter(dcd_filename, 10)  # save every 10 steps
    simulation.reporters.append(dcd)

    total_steps = 1000  # very short for testing
    step_size = 10
    bias_list = []
    for i in range(total_steps // step_size):
        meta.step(simulation, step_size)
        simulation.step(0)  # triggers reporters
        # Save the current bias value for each frame in this chunk
        # Use meta.getBiasVariables() or meta.getBiasForWalker() if available, else meta._currentBias
        # For now, we will use meta.getBiasVariables()[0] if available
        try:
            # This is a placeholder; you may need to adjust depending on your OpenMM version
            bias_val = meta._currentBias
        except AttributeError:
            bias_val = 0.0
        for _ in range(step_size):
            bias_list.append(bias_val)

    simulation.saveState("final.xml")
    print("\u2714 MD + biasing finished\n")

    # Remove DCDReporter and force garbage collection to finalize file
    simulation.reporters.remove(dcd)
    import gc
    del dcd
    gc.collect()

    # Save the bias array for this run
    bias_array = np.array(bias_list)
    bias_file = BASE_DIR / "bias" / "bias_for_run.npy"
    np.save(str(bias_file), bias_array)
    print(f"[INFO] Saved bias array for this run to {bias_file} (length: {len(bias_array)})")

def feature_extraction(dcd_path, pdb_path):
    print("Stage 4/5  –  featurisation + clustering ...")

    # Load the trajectory and compute φ dihedral angles
    t = md.load(dcd_path, top=pdb_path)
    print("Number of frames loaded:", t.n_frames)
    phi_vals, _ = md.compute_phi(t)
    phi_vals = phi_vals.squeeze()
    X = np.cos(phi_vals)
    X = X.reshape(-1, 1)

    kmeans = MiniBatchKMeans(n_clusters=40, random_state=0).fit(X)
    states = kmeans.labels_
    print("\u2714 Clustering done\n")
    return states

def build_transition_model(states, bias=None):
    print("Stage 5/5  –  Markov model ...")

    tau = 20  # frames → 40 ps
    C = defaultdict(float)
    kT = 0.593  # kcal/mol at 300K
    F_est = 0.0  # For now, can be improved later
    n_transitions = len(states) - tau
    if bias is not None and len(bias) != len(states):
        raise ValueError(f"Bias array length ({len(bias)}) does not match number of states ({len(states)})")
    for i in range(n_transitions):
        if bias is not None:
            w_t = np.exp((bias[i] - F_est) / kT)
        else:
            w_t = 1.0
        C[(states[i], states[i + tau])] += w_t

    # Dense count matrix → row-normalised transition matrix
    n = np.max(states) + 1
    Cmat = np.zeros((n, n))
    for (i, j), w in C.items():
        Cmat[i, j] = w

    T = (Cmat.T / Cmat.sum(axis=1)).T  # row-stochastic

    # Stationary distribution (left eigenvector of T)
    evals, evecs = np.linalg.eig(T.T)
    pi = np.real_if_close(evecs[:, np.argmax(evals)].flatten())
    pi /= pi.sum()
    DG = -kT * np.log(pi)  # 0.593 kcal/mol ≈ kT at 300 K

    print("\u2714 Finished – free energies (kcal/mol) written to DG array")
    return DG

def relative_energies(DG):
    return DG - np.min(DG)

def plot_DG(DG):
    plt.figure()
    plt.bar(np.arange(len(DG)), DG, color='blue')
    plt.xlabel('State Index')
    plt.ylabel('Free Energy (kcal/mol)')
    plt.title('Free Energy Profile')
    plt.tight_layout()
    plt.show()

