"""
Enhanced Markov State Model analysis with TRAM/dTRAM and comprehensive reporting.

This module provides advanced MSM analysis capabilities including:
- TRAM/dTRAM for multi-temperature data
- Free energy surface generation
- State table export
- Implied timescales analysis
- Representative structure extraction
- Comprehensive visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mdtraj as md
from pathlib import Path
import pickle
from typing import List, Tuple, Optional, Dict, Any, Union
import logging
from scipy import constants
from scipy.constants import Boltzmann as kB
from scipy.sparse import save_npz, load_npz, issparse, csc_matrix
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)


class EnhancedMSM:
    """
    Enhanced Markov State Model with advanced analysis and reporting capabilities.
    
    This class provides comprehensive MSM analysis including multi-temperature
    data handling, free energy surface generation, and detailed reporting.
    """
    
    def __init__(self, 
                 trajectory_files: Union[str, List[str]] = None,
                 topology_file: str = None,
                 temperatures: List[float] = None,
                 output_dir: str = "msm_analysis"):
        """
        Initialize the Enhanced MSM analyzer.
        
        Args:
            trajectory_files: Single trajectory file or list of files
            topology_file: Topology file (PDB) for the system
            temperatures: List of temperatures for TRAM analysis
            output_dir: Directory for output files
        """
        self.trajectory_files = trajectory_files if isinstance(trajectory_files, list) else [trajectory_files] if trajectory_files else []
        self.topology_file = topology_file
        self.temperatures = temperatures or [300.0]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Analysis data
        self.trajectories = []
        self.dtrajs = []  # Discrete trajectories
        self.features = None
        self.cluster_centers = None
        self.n_states = 0
        
        # MSM data
        self.transition_matrix = None
        self.count_matrix = None
        self.stationary_distribution = None
        self.free_energies = None
        self.lag_time = 20  # Default lag time
        
        # TRAM data
        self.tram_weights = None
        self.multi_temp_counts = {}
        
        # Analysis results
        self.implied_timescales = None
        self.state_table = None
        self.fes_data = None
        
        logger.info(f"Enhanced MSM initialized for {len(self.trajectory_files)} trajectories")
    
    def load_trajectories(self, stride: int = 1):
        """
        Load trajectory data for analysis.
        
        Args:
            stride: Stride for loading frames (1 = every frame)
        """
        logger.info("Loading trajectory data...")
        
        self.trajectories = []
        for i, traj_file in enumerate(self.trajectory_files):
            if Path(traj_file).exists():
                traj = md.load(traj_file, top=self.topology_file, stride=stride)
                self.trajectories.append(traj)
                logger.info(f"Loaded trajectory {i+1}: {traj.n_frames} frames")
            else:
                logger.warning(f"Trajectory file not found: {traj_file}")
        
        if not self.trajectories:
            raise ValueError("No trajectories loaded successfully")
        
        logger.info(f"Total trajectories loaded: {len(self.trajectories)}")
    
    def compute_features(self, feature_type: str = "phi_psi", n_features: int = None):
        """
        Compute features from trajectory data.
        
        Args:
            feature_type: Type of features to compute ('phi_psi', 'distances', 'contacts')
            n_features: Number of features to compute (auto if None)
        """
        logger.info(f"Computing {feature_type} features...")
        
        all_features = []
        
        for traj in self.trajectories:
            if feature_type == "phi_psi":
                phi_angles, _ = md.compute_phi(traj)
                psi_angles, _ = md.compute_psi(traj)
                
                # Convert to cosine/sine representation for periodicity
                features = []
                if phi_angles.shape[1] > 0:
                    features.extend([np.cos(phi_angles), np.sin(phi_angles)])
                if psi_angles.shape[1] > 0:
                    features.extend([np.cos(psi_angles), np.sin(psi_angles)])
                
                if features:
                    traj_features = np.hstack(features)
                else:
                    logger.warning("No dihedral angles found, using Cartesian coordinates")
                    traj_features = traj.xyz.reshape(traj.n_frames, -1)
            
            elif feature_type == "distances":
                # Compute all Cα-Cα distances
                ca_indices = traj.topology.select("name CA")
                if len(ca_indices) < 2:
                    raise ValueError("Insufficient Cα atoms for distance features")
                
                # Select pairs (every 3rd residue to reduce dimensionality)
                if n_features:
                    n_pairs = min(n_features, len(ca_indices) * (len(ca_indices) - 1) // 2)
                else:
                    n_pairs = min(200, len(ca_indices) * (len(ca_indices) - 1) // 2)
                
                pairs = []
                for i in range(0, len(ca_indices), 3):
                    for j in range(i + 3, len(ca_indices), 3):
                        pairs.append([ca_indices[i], ca_indices[j]])
                        if len(pairs) >= n_pairs:
                            break
                    if len(pairs) >= n_pairs:
                        break
                
                traj_features = md.compute_distances(traj, pairs)
            
            elif feature_type == "contacts":
                # Compute native contacts
                ca_indices = traj.topology.select("name CA")
                contacts, pairs = md.compute_contacts(traj, contacts='all', scheme='ca')
                traj_features = contacts
            
            else:
                raise ValueError(f"Unknown feature type: {feature_type}")
            
            all_features.append(traj_features)
        
        # Combine all features
        self.features = np.vstack(all_features)
        logger.info(f"Features computed: {self.features.shape}")
    
    def cluster_features(self, n_clusters: int = 100, algorithm: str = "kmeans"):
        """
        Cluster features to create discrete states.
        
        Args:
            n_clusters: Number of clusters (states)
            algorithm: Clustering algorithm ('kmeans', 'gmm')
        """
        logger.info(f"Clustering features into {n_clusters} states using {algorithm}...")
        
        if self.features is None:
            raise ValueError("Features must be computed before clustering")
        
        if algorithm == "kmeans":
            clusterer = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
            labels = clusterer.fit_predict(self.features)
            self.cluster_centers = clusterer.cluster_centers_
        else:
            raise ValueError(f"Clustering algorithm {algorithm} not implemented")
        
        # Split labels back into trajectories
        self.dtrajs = []
        start_idx = 0
        for traj in self.trajectories:
            end_idx = start_idx + traj.n_frames
            self.dtrajs.append(labels[start_idx:end_idx])
            start_idx = end_idx
        
        self.n_states = n_clusters
        logger.info(f"Clustering completed: {n_clusters} states")
    
    def build_msm(self, lag_time: int = 20, method: str = "standard"):
        """
        Build Markov State Model from discrete trajectories.
        
        Args:
            lag_time: Lag time for transition counting
            method: MSM method ('standard', 'tram')
        """
        logger.info(f"Building MSM with lag time {lag_time} using {method} method...")
        
        self.lag_time = lag_time
        
        if method == "standard":
            self._build_standard_msm(lag_time)
        elif method == "tram":
            self._build_tram_msm(lag_time)
        else:
            raise ValueError(f"Unknown MSM method: {method}")
        
        # Compute free energies
        self._compute_free_energies()
        
        logger.info("MSM construction completed")
    
    def _build_standard_msm(self, lag_time: int):
        """Build standard MSM from single temperature data."""
        # Count transitions
        counts = defaultdict(float)
        total_transitions = 0
        
        for dtraj in self.dtrajs:
            for i in range(len(dtraj) - lag_time):
                state_i = dtraj[i]
                state_j = dtraj[i + lag_time]
                counts[(state_i, state_j)] += 1.0
                total_transitions += 1
        
        # Build count matrix
        count_matrix = np.zeros((self.n_states, self.n_states))
        for (i, j), count in counts.items():
            count_matrix[i, j] = count
        
        self.count_matrix = count_matrix
        
        # Build transition matrix (row-stochastic)
        row_sums = count_matrix.sum(axis=1)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        self.transition_matrix = count_matrix / row_sums[:, np.newaxis]
        
        # Compute stationary distribution
        eigenvals, eigenvecs = np.linalg.eig(self.transition_matrix.T)
        stationary_idx = np.argmax(np.real(eigenvals))
        stationary = np.real(eigenvecs[:, stationary_idx])
        stationary = stationary / stationary.sum()
        self.stationary_distribution = np.abs(stationary)  # Ensure positive
    
    def _build_tram_msm(self, lag_time: int):
        """Build MSM using TRAM for multi-temperature data."""
        logger.info("Building TRAM MSM for multi-temperature data...")
        
        # This is a simplified TRAM implementation
        # For production use, consider using packages like pyemma or deeptime
        
        if len(self.temperatures) == 1:
            logger.warning("Only one temperature provided, falling back to standard MSM")
            return self._build_standard_msm(lag_time)
        
        # Count transitions for each temperature
        temp_counts = {}
        for temp_idx, temp in enumerate(self.temperatures):
            if temp_idx < len(self.dtrajs):
                dtraj = self.dtrajs[temp_idx]
                counts = defaultdict(float)
                
                for i in range(len(dtraj) - lag_time):
                    state_i = dtraj[i]
                    state_j = dtraj[i + lag_time]
                    counts[(state_i, state_j)] += 1.0
                
                temp_counts[temp] = counts
        
        # Simplified TRAM: weight by Boltzmann factors
        # This is a basic implementation - real TRAM is more sophisticated
        kT_ref = constants.k * 300.0  # Reference temperature
        
        combined_counts = defaultdict(float)
        for temp, counts in temp_counts.items():
            kT = constants.k * temp
            weight = kT_ref / kT  # Simple reweighting
            
            for (i, j), count in counts.items():
                combined_counts[(i, j)] += count * weight
        
        # Build matrices from combined counts
        count_matrix = np.zeros((self.n_states, self.n_states))
        for (i, j), count in combined_counts.items():
            count_matrix[i, j] = count
        
        self.count_matrix = count_matrix
        
        # Build transition matrix
        row_sums = count_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1
        self.transition_matrix = count_matrix / row_sums[:, np.newaxis]
        
        # Compute stationary distribution
        eigenvals, eigenvecs = np.linalg.eig(self.transition_matrix.T)
        stationary_idx = np.argmax(np.real(eigenvals))
        stationary = np.real(eigenvecs[:, stationary_idx])
        stationary = stationary / stationary.sum()
        self.stationary_distribution = np.abs(stationary)
    
    def _compute_free_energies(self, temperature: float = 300.0):
        """Compute free energies from stationary distribution."""
        kT = constants.k * temperature * constants.Avogadro / 1000.0  # kJ/mol
        
        # Avoid log(0) by adding small epsilon
        pi_safe = np.maximum(self.stationary_distribution, 1e-12)
        self.free_energies = -kT * np.log(pi_safe)
        
        # Set relative to minimum
        self.free_energies -= np.min(self.free_energies)
        
        logger.info(f"Free energies computed (range: 0 - {np.max(self.free_energies):.2f} kJ/mol)")
    
    def compute_implied_timescales(self, lag_times: List[int] = None, n_timescales: int = 5):
        """
        Compute implied timescales for MSM validation.
        
        Args:
            lag_times: List of lag times to test
            n_timescales: Number of timescales to compute
        """
        logger.info("Computing implied timescales...")
        
        if lag_times is None:
            lag_times = list(range(1, 101, 5))  # 1 to 100 in steps of 5
        
        timescales_data = []
        
        for lag in lag_times:
            try:
                # Build MSM for this lag time
                self._build_standard_msm(lag)
                
                # Compute eigenvalues
                eigenvals = np.linalg.eigvals(self.transition_matrix)
                eigenvals = np.real(eigenvals)
                eigenvals = np.sort(eigenvals)[::-1]  # Sort descending
                
                # Convert to timescales (excluding the stationary eigenvalue)
                timescales = []
                for i in range(1, min(n_timescales + 1, len(eigenvals))):
                    if eigenvals[i] > 0 and eigenvals[i] < 1:
                        ts = -lag / np.log(eigenvals[i])
                        timescales.append(ts)
                
                # Pad with NaN if not enough timescales
                while len(timescales) < n_timescales:
                    timescales.append(np.nan)
                
                timescales_data.append(timescales[:n_timescales])
                
            except Exception as e:
                logger.warning(f"Failed to compute timescales for lag {lag}: {e}")
                timescales_data.append([np.nan] * n_timescales)
        
        self.implied_timescales = {
            'lag_times': lag_times,
            'timescales': np.array(timescales_data)
        }
        
        # Restore original MSM
        self._build_standard_msm(self.lag_time)
        
        logger.info("Implied timescales computation completed")
    
    def generate_free_energy_surface(self, 
                                   cv1_name: str = "phi", 
                                   cv2_name: str = "psi",
                                   bins: int = 50,
                                   temperature: float = 300.0) -> Dict[str, np.ndarray]:
        """
        Generate 2D free energy surface from MSM data.
        
        Args:
            cv1_name: Name of first collective variable
            cv2_name: Name of second collective variable
            bins: Number of bins for 2D histogram
            temperature: Temperature for free energy calculation
            
        Returns:
            Dictionary containing FES data
        """
        logger.info(f"Generating free energy surface: {cv1_name} vs {cv2_name}")
        
        if self.features is None or self.stationary_distribution is None:
            raise ValueError("Features and MSM must be computed first")
        
        # Map features to collective variables
        cv1_data, cv2_data = self._extract_collective_variables(cv1_name, cv2_name)
        
        # Map stationary distribution to frames
        frame_weights = []
        for dtraj in self.dtrajs:
            for state in dtraj:
                frame_weights.append(self.stationary_distribution[state])
        
        frame_weights = np.array(frame_weights)
        
        # Validate data sufficiency for meaningful analysis
        total_frames = len(frame_weights)
        cv_points = len(cv1_data)
        
        logger.info(f"MSM Analysis data: {cv_points} CV points, {total_frames} trajectory frames")
        
        # Check if we have sufficient data for meaningful free energy surface
        min_frames_required = bins // 4  # At least bins/4 frames for meaningful histogram
        if total_frames < min_frames_required:
            logger.warning(f"Insufficient data for {bins}x{bins} FES: {total_frames} frames < {min_frames_required} required")
            logger.info(f"Reducing bins from {bins} to {max(10, total_frames // 2)} for sparse data")
            bins = max(10, total_frames // 2)
        
        # Ensure weights and data arrays have the same length
        min_length = min(len(cv1_data), len(cv2_data), len(frame_weights))
        if len(cv1_data) != len(frame_weights):
            logger.warning(f"Length mismatch: CV data ({len(cv1_data)}) vs weights ({len(frame_weights)}). "
                         f"Truncating to {min_length} points.")
            cv1_data = cv1_data[:min_length]
            cv2_data = cv2_data[:min_length]
            frame_weights = frame_weights[:min_length]
        
        # Create 2D histogram weighted by stationary probabilities
        try:
            H, xedges, yedges = np.histogram2d(
                cv1_data, cv2_data, 
                bins=bins, 
                weights=frame_weights, 
                density=True
            )
            
            # Check for sufficient non-zero bins
            non_zero_bins = np.sum(H > 0)
            logger.info(f"Histogram: {non_zero_bins} non-zero bins out of {bins*bins} total")
            
            if non_zero_bins < 3:
                logger.warning("Very sparse histogram - results may not be meaningful")
                
        except Exception as e:
            logger.error(f"Histogram generation failed: {e}")
            raise ValueError(f"Could not generate histogram for FES: {e}")
        
        # Convert to free energy
        kT = constants.k * temperature * constants.Avogadro / 1000.0  # kJ/mol
        
        # Handle sparse histograms more carefully
        # Only convert non-zero bins to free energy
        F = np.full_like(H, np.inf)  # Initialize with inf
        mask = H > 1e-12  # Only bins with reasonable probability
        
        if np.sum(mask) == 0:
            logger.error("No populated bins in histogram - cannot generate meaningful FES")
            raise ValueError("Histogram too sparse for free energy calculation. "
                           "Try: 1) Longer simulation, 2) Fewer bins, 3) Different CVs")
        
        # Calculate free energy only for populated bins
        F[mask] = -kT * np.log(H[mask])
        
        # Set relative to minimum (only consider finite values)
        finite_mask = np.isfinite(F)
        if np.sum(finite_mask) == 0:
            logger.error("No finite free energy values - calculation failed")
            raise ValueError("All free energy values are infinite - histogram too sparse")
            
        F_min = np.min(F[finite_mask])
        F[finite_mask] -= F_min
        
        # Log statistics about the FES
        n_finite = np.sum(finite_mask)
        n_total = H.size
        logger.info(f"Free energy surface: {n_finite}/{n_total} finite bins, "
                   f"range: {F_min:.2f} to {np.max(F[finite_mask]):.2f} kJ/mol")
        
        # Store FES data
        self.fes_data = {
            'free_energy': F,
            'xedges': xedges,
            'yedges': yedges,
            'cv1_name': cv1_name,
            'cv2_name': cv2_name,
            'temperature': temperature
        }
        
        logger.info("Free energy surface generated")
        return self.fes_data
    
    def _extract_collective_variables(self, cv1_name: str, cv2_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Extract collective variables from trajectory data."""
        cv1_data = []
        cv2_data = []
        
        for traj in self.trajectories:
            if cv1_name == "phi" and cv2_name == "psi":
                phi_angles, _ = md.compute_phi(traj)
                psi_angles, _ = md.compute_psi(traj)
                
                if phi_angles.shape[1] > 0 and psi_angles.shape[1] > 0:
                    cv1_data.extend(phi_angles[:, 0])  # First phi angle
                    cv2_data.extend(psi_angles[:, 0])  # First psi angle
                else:
                    raise ValueError("No phi/psi angles found in trajectory")
            
            elif "distance" in cv1_name or "distance" in cv2_name:
                # Simple distance features
                ca_indices = traj.topology.select("name CA")
                if len(ca_indices) >= 4:
                    dist1 = md.compute_distances(traj, [[ca_indices[0], ca_indices[-1]]])
                    dist2 = md.compute_distances(traj, [[ca_indices[len(ca_indices)//2], ca_indices[-1]]])
                    cv1_data.extend(dist1.flatten())
                    cv2_data.extend(dist2.flatten())
                else:
                    raise ValueError("Insufficient atoms for distance calculation")
            
            else:
                # Use first two principal components of features
                if self.features.shape[1] >= 2:
                    start_idx = sum(t.n_frames for t in self.trajectories[:self.trajectories.index(traj)])
                    end_idx = start_idx + traj.n_frames
                    cv1_data.extend(self.features[start_idx:end_idx, 0])
                    cv2_data.extend(self.features[start_idx:end_idx, 1])
                else:
                    raise ValueError("Insufficient feature dimensions")
        
        return np.array(cv1_data), np.array(cv2_data)
    
    def create_state_table(self) -> pd.DataFrame:
        """Create comprehensive state summary table."""
        logger.info("Creating state summary table...")
        
        if self.stationary_distribution is None:
            raise ValueError("MSM must be built before creating state table")
        
        # Basic state information
        state_data = {
            'state_id': range(self.n_states),
            'population': self.stationary_distribution,
            'free_energy_kJ_mol': self.free_energies,
            'free_energy_kcal_mol': self.free_energies * 0.239006  # Convert kJ/mol to kcal/mol
        }
        
        # Count frames per state
        frame_counts = np.zeros(self.n_states)
        total_frames = 0
        for dtraj in self.dtrajs:
            for state in dtraj:
                frame_counts[state] += 1
                total_frames += 1
        
        state_data['frame_count'] = frame_counts.astype(int)
        state_data['frame_percentage'] = 100 * frame_counts / total_frames
        
        # Find representative frames (centroid of each state)
        representative_frames = []
        centroid_features = []
        
        for state in range(self.n_states):
            # Find all frames in this state
            state_frames = []
            state_features = []
            
            frame_idx = 0
            for traj_idx, dtraj in enumerate(self.dtrajs):
                for local_frame, assigned_state in enumerate(dtraj):
                    if assigned_state == state:
                        state_frames.append((traj_idx, local_frame))
                        if self.features is not None:
                            state_features.append(self.features[frame_idx])
                    frame_idx += 1
            
            if state_features:
                # Find centroid frame
                state_features = np.array(state_features)
                centroid = np.mean(state_features, axis=0)
                
                # Find closest frame to centroid
                distances = np.linalg.norm(state_features - centroid, axis=1)
                closest_idx = np.argmin(distances)
                representative_frames.append(state_frames[closest_idx])
                centroid_features.append(centroid)
            else:
                representative_frames.append((-1, -1))  # No frames in this state
                centroid_features.append(None)
        
        state_data['representative_traj'] = [int(rf[0]) for rf in representative_frames]
        state_data['representative_frame'] = [int(rf[1]) for rf in representative_frames]
        
        # Add cluster center information if available
        if self.cluster_centers is not None:
            for i, center in enumerate(self.cluster_centers.T):
                state_data[f'cluster_center_{i}'] = center
        
        self.state_table = pd.DataFrame(state_data)
        
        logger.info(f"State table created with {len(self.state_table)} states")
        return self.state_table
    
    def _create_matrix_intelligent(self, shape: Tuple[int, int], use_sparse: bool = None) -> Union[np.ndarray, csc_matrix]:
        """
        Intelligently create matrix (sparse or dense) based on expected size and sparsity.
        
        Args:
            shape: Matrix shape (n_states, n_states)
            use_sparse: Force sparse (True) or dense (False). If None, auto-decide.
            
        Returns:
            Zero matrix of appropriate type
        """
        n_states = shape[0]
        
        if use_sparse is None:
            # Auto-decide based on size - sparse for large state spaces
            use_sparse = n_states > 100
        
        if use_sparse:
            logger.debug(f"Creating sparse matrix ({n_states}x{n_states})")
            return csc_matrix(shape, dtype=np.float64)
        else:
            logger.debug(f"Creating dense matrix ({n_states}x{n_states})")
            return np.zeros(shape, dtype=np.float64)

    def _matrix_add_count(self, matrix: Union[np.ndarray, csc_matrix], i: int, j: int, count: float):
        """Add count to matrix element, handling both sparse and dense matrices."""
        if issparse(matrix):
            matrix[i, j] += count
        else:
            matrix[i, j] += count
    
    def _matrix_normalize_rows(self, matrix: Union[np.ndarray, csc_matrix]) -> Union[np.ndarray, csc_matrix]:
        """Normalize matrix rows to create transition matrix, handling both sparse and dense."""
        if issparse(matrix):
            # Sparse matrix row normalization
            row_sums = np.array(matrix.sum(axis=1)).flatten()
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            row_diag = csc_matrix((1.0 / row_sums, (range(len(row_sums)), range(len(row_sums)))), 
                                shape=(len(row_sums), len(row_sums)))
            return row_diag @ matrix
        else:
            # Dense matrix row normalization
            row_sums = matrix.sum(axis=1)
            row_sums[row_sums == 0] = 1
            return matrix / row_sums[:, np.newaxis]

    def _save_matrix_intelligent(self, matrix, filename_base: str, prefix: str = "msm_analysis"):
        """
        Intelligently save matrix in appropriate format(s) based on type and size.
        
        Args:
            matrix: numpy array or scipy sparse matrix
            filename_base: base filename (e.g., "transition_matrix")
            prefix: file prefix
        """
        if matrix is None:
            return
            
        # Always save dense format for compatibility
        np.save(self.output_dir / f"{prefix}_{filename_base}.npy", 
                matrix.toarray() if issparse(matrix) else matrix)
        
        # For large matrices, also save sparse format for efficiency
        if matrix.size > 10000:  # >100x100 matrix
            if issparse(matrix):
                # Already sparse - save directly
                save_npz(self.output_dir / f"{prefix}_{filename_base}.npz", matrix)
            else:
                # Convert dense to sparse if beneficial
                # Only convert if matrix is actually sparse (>95% zeros)
                sparsity = np.count_nonzero(matrix) / matrix.size
                if sparsity < 0.05:  # Less than 5% non-zero
                    sparse_matrix = csc_matrix(matrix)
                    save_npz(self.output_dir / f"{prefix}_{filename_base}.npz", sparse_matrix)
                    logger.info(f"Converted {filename_base} to sparse format (sparsity: {(1-sparsity)*100:.1f}% zeros)")
                else:
                    logger.debug(f"Keeping {filename_base} as dense (sparsity: {(1-sparsity)*100:.1f}% zeros)")

    def save_analysis_results(self, prefix: str = "msm_analysis"):
        """Save all analysis results to files."""
        logger.info("Saving analysis results...")
        
        # Save transition matrix with intelligent format selection
        self._save_matrix_intelligent(self.transition_matrix, "transition_matrix", prefix)
        
        # Save count matrix with intelligent format selection
        self._save_matrix_intelligent(self.count_matrix, "count_matrix", prefix)
        
        # Save free energies
        if self.free_energies is not None:
            np.save(self.output_dir / f"{prefix}_free_energies.npy", self.free_energies)
        
        # Save stationary distribution
        if self.stationary_distribution is not None:
            np.save(self.output_dir / f"{prefix}_stationary_distribution.npy", 
                   self.stationary_distribution)
        
        # Save discrete trajectories
        if self.dtrajs:
            np.save(self.output_dir / f"{prefix}_dtrajs.npy", self.dtrajs)
        
        # Save state table
        if self.state_table is not None:
            self.state_table.to_csv(self.output_dir / f"{prefix}_state_table.csv", index=False)
        
        # Save FES data
        if self.fes_data is not None:
            np.save(self.output_dir / f"{prefix}_fes.npy", self.fes_data['free_energy'])
            
            # Save FES metadata
            fes_metadata = {k: v for k, v in self.fes_data.items() if k != 'free_energy'}
            with open(self.output_dir / f"{prefix}_fes_metadata.pkl", 'wb') as f:
                pickle.dump(fes_metadata, f)
        
        # Save implied timescales
        if self.implied_timescales is not None:
            with open(self.output_dir / f"{prefix}_implied_timescales.pkl", 'wb') as f:
                pickle.dump(self.implied_timescales, f)
        
        logger.info(f"Analysis results saved to {self.output_dir}")
    
    def plot_free_energy_surface(self, save_file: str = None, interactive: bool = False):
        """Plot the free energy surface."""
        if self.fes_data is None:
            raise ValueError("Free energy surface must be generated first")
        
        F = self.fes_data['free_energy']
        xedges = self.fes_data['xedges']
        yedges = self.fes_data['yedges']
        cv1_name = self.fes_data['cv1_name']
        cv2_name = self.fes_data['cv2_name']
        
        if interactive:
            try:
                import plotly.graph_objects as go
                
                # Create interactive plot
                x_centers = 0.5 * (xedges[:-1] + xedges[1:])
                y_centers = 0.5 * (yedges[:-1] + yedges[1:])
                
                fig = go.Figure(data=go.Contour(
                    z=F.T,
                    x=x_centers,
                    y=y_centers,
                    colorscale='viridis',
                    colorbar=dict(title="Free Energy (kJ/mol)")
                ))
                
                fig.update_layout(
                    title=f"Free Energy Surface ({cv1_name} vs {cv2_name})",
                    xaxis_title=cv1_name,
                    yaxis_title=cv2_name
                )
                
                if save_file:
                    fig.write_html(str(self.output_dir / f"{save_file}.html"))
                
                fig.show()
                
            except ImportError:
                logger.warning("Plotly not available, falling back to matplotlib")
                interactive = False
        
        if not interactive:
            # Create matplotlib plot
            plt.figure(figsize=(10, 8))
            
            x_centers = 0.5 * (xedges[:-1] + xedges[1:])
            y_centers = 0.5 * (yedges[:-1] + yedges[1:])
            
            contour = plt.contourf(x_centers, y_centers, F.T, levels=20, cmap='viridis')
            plt.colorbar(contour, label='Free Energy (kJ/mol)')
            
            plt.xlabel(cv1_name)
            plt.ylabel(cv2_name)
            plt.title(f'Free Energy Surface ({cv1_name} vs {cv2_name})')
            
            if save_file:
                plt.savefig(self.output_dir / f"{save_file}.png", dpi=300, bbox_inches='tight')
            
            plt.show()
    
    def plot_implied_timescales(self, save_file: str = None):
        """Plot implied timescales for MSM validation."""
        if self.implied_timescales is None:
            raise ValueError("Implied timescales must be computed first")
        
        lag_times = self.implied_timescales['lag_times']
        timescales = self.implied_timescales['timescales']
        
        plt.figure(figsize=(10, 6))
        
        for i in range(timescales.shape[1]):
            plt.plot(lag_times, timescales[:, i], 'o-', label=f'Timescale {i+1}')
        
        plt.xlabel('Lag Time')
        plt.ylabel('Implied Timescale')
        plt.title('Implied Timescales Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_file:
            plt.savefig(self.output_dir / f"{save_file}.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_free_energy_profile(self, save_file: str = None):
        """Plot 1D free energy profile by state."""
        if self.free_energies is None:
            raise ValueError("Free energies must be computed first")
        
        plt.figure(figsize=(12, 6))
        
        state_ids = np.arange(len(self.free_energies))
        plt.bar(state_ids, self.free_energies, alpha=0.7, color='steelblue')
        
        plt.xlabel('State Index')
        plt.ylabel('Free Energy (kJ/mol)')
        plt.title('Free Energy Profile by State')
        plt.grid(True, alpha=0.3)
        
        if save_file:
            plt.savefig(self.output_dir / f"{save_file}.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def extract_representative_structures(self, save_pdb: bool = True):
        """Extract and optionally save representative structures for each state."""
        logger.info("Extracting representative structures...")
        
        if self.state_table is None:
            self.create_state_table()
        
        representative_structures = []
        
        for _, row in self.state_table.iterrows():
            try:
                traj_idx = int(row['representative_traj'])
                frame_idx = int(row['representative_frame']) 
                state_id = int(row['state_id'])
                
                if traj_idx >= 0 and frame_idx >= 0:
                    # Validate indices
                    if traj_idx >= len(self.trajectories):
                        logger.warning(f"Invalid trajectory index {traj_idx} for state {state_id}")
                        continue
                        
                    traj = self.trajectories[traj_idx]
                    if frame_idx >= len(traj):
                        logger.warning(f"Invalid frame index {frame_idx} for state {state_id}")
                        continue
                        
                    # Extract frame
                    frame = traj[frame_idx]
                    representative_structures.append((state_id, frame))
                    
                    if save_pdb:
                        output_file = self.output_dir / f"state_{state_id:03d}_representative.pdb"
                        frame.save_pdb(str(output_file))
                        
            except (ValueError, TypeError, IndexError) as e:
                logger.warning(f"Error extracting representative structure for state {state_id}: {e}")
                continue
        
        logger.info(f"Extracted {len(representative_structures)} representative structures")
        return representative_structures


# Convenience function for complete analysis pipeline
def run_complete_msm_analysis(trajectory_files: Union[str, List[str]],
                            topology_file: str,
                            output_dir: str = "msm_analysis",
                            n_clusters: int = 100,
                            lag_time: int = 20,
                            feature_type: str = "phi_psi",
                            temperatures: List[float] = None) -> EnhancedMSM:
    """
    Run complete MSM analysis pipeline.
    
    Args:
        trajectory_files: Trajectory file(s) to analyze
        topology_file: Topology file (PDB)
        output_dir: Output directory
        n_clusters: Number of states for clustering
        lag_time: Lag time for MSM construction
        feature_type: Type of features to use
        temperatures: Temperatures for TRAM analysis
        
    Returns:
        EnhancedMSM object with completed analysis
    """
    # Initialize analyzer
    msm = EnhancedMSM(
        trajectory_files=trajectory_files,
        topology_file=topology_file,
        temperatures=temperatures,
        output_dir=output_dir
    )
    
    # Load trajectories
    msm.load_trajectories()
    
    # Compute features and cluster
    msm.compute_features(feature_type=feature_type)
    msm.cluster_features(n_clusters=n_clusters)
    
    # Build MSM
    method = "tram" if temperatures and len(temperatures) > 1 else "standard"
    msm.build_msm(lag_time=lag_time, method=method)
    
    # Compute implied timescales
    msm.compute_implied_timescales()
    
    # Generate free energy surface (with graceful handling for small datasets)
    fes_success = False
    try:
        if feature_type == "phi_psi":
            msm.generate_free_energy_surface(cv1_name="phi", cv2_name="psi")
        else:
            msm.generate_free_energy_surface(cv1_name="CV1", cv2_name="CV2")
        fes_success = True
        logger.info("✓ Free energy surface generation completed")
    except ValueError as e:
        logger.warning(f"⚠ Free energy surface generation failed: {e}")
        logger.info("Continuing with analysis without FES plots...")
        
    # Create state table
    msm.create_state_table()
    
    # Extract representative structures
    msm.extract_representative_structures()
    
    # Save all results
    msm.save_analysis_results()
    
    # Generate plots (skip FES plots if generation failed)
    if fes_success:
        try:
            msm.plot_free_energy_surface(save_file="free_energy_surface")
            logger.info("✓ Free energy surface plot saved")
        except Exception as e:
            logger.warning(f"⚠ Free energy surface plotting failed: {e}")
    else:
        logger.info("⚠ Skipping free energy surface plots due to insufficient data")
        
    # Always try these plots as they don't depend on FES
    try:
        msm.plot_implied_timescales(save_file="implied_timescales")
        logger.info("✓ Implied timescales plot saved")
    except Exception as e:
        logger.warning(f"⚠ Implied timescales plotting failed: {e}")
        
    try:
        msm.plot_free_energy_profile(save_file="free_energy_profile")
        logger.info("✓ Free energy profile plot saved")
    except Exception as e:
        logger.warning(f"⚠ Free energy profile plotting failed: {e}")
    
    logger.info("Complete MSM analysis finished")
    return msm