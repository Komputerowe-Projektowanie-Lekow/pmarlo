#!/usr/bin/env python3
"""
Utility script to export CV bias bundle from an existing trained DeepTICA model.

Usage:
    python export_cv_bundle.py <model_base_path>

Example:
    python export_cv_bundle.py app_output/models/deeptica-20251108-175716

This will create the deeptica_cv_model.* files needed for metabiased simulation.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from pmarlo.features.deeptica.ts_feature_extractor import canonicalize_feature_spec

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def export_cv_bundle_from_model(model_base_path: Path, output_dir: Path = None) -> None:
    """
    Export CV bias bundle from a trained DeepTICA model.
    
    Parameters
    ----------
    model_base_path : Path
        Path to the model base (without suffix), e.g., deeptica-20251108-175716
    output_dir : Path, optional
        Output directory for the CV bundle. If None, uses the model's directory.
    """
    try:
        # Import required modules
        from pmarlo.features.deeptica import export_cv_model
        from pmarlo.features.deeptica._full import DeepTICAModel
        import yaml
        
        model_base_path = Path(model_base_path)
        
        # Check if model files exist
        pt_file = model_base_path.with_suffix(".pt")
        scaler_file = model_base_path.with_suffix(".scaler.pt")
        config_file = model_base_path.with_suffix(".json")
        
        if not pt_file.exists():
            raise FileNotFoundError(f"Model file not found: {pt_file}")
        if not scaler_file.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_file}")
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        logger.info(f"Loading DeepTICA model from {model_base_path}")
        
        # Load the full DeepTICA model
        try:
            deeptica_model = DeepTICAModel.load(model_base_path)
            network = deeptica_model.net
            scaler = deeptica_model.scaler
            history = deeptica_model.training_history or {}
        except Exception as load_error:
            logger.warning(f"Standard model loading failed: {load_error}")
            logger.info("Attempting alternative loading method...")
            
            # Alternative loading: manually construct the model from saved components
            import json
            import torch
            from sklearn.preprocessing import StandardScaler
            
            # Load config
            config_data = json.loads(config_file.read_text(encoding="utf-8"))
            logger.info(f"Loaded config: {config_data}")
            
            # Load scaler
            scaler_ckpt = torch.load(scaler_file, map_location="cpu", weights_only=False)
            scaler = StandardScaler(with_mean=True, with_std=True)
            scaler.mean_ = scaler_ckpt["mean"]
            scaler.scale_ = scaler_ckpt["std"]
            logger.info(f"Loaded scaler: mean shape={scaler.mean_.shape}, scale shape={scaler.scale_.shape}")
            
            # Load network state dict
            state_dict = torch.load(pt_file, map_location="cpu", weights_only=False)
            logger.info(f"Loaded state dict with keys: {list(state_dict.keys())}")
            
            # Extract the actual network from state_dict
            if "state_dict" in state_dict:
                actual_state = state_dict["state_dict"]
            else:
                actual_state = state_dict
            
            # Create a simple wrapper that exposes the network and scaler
            from pmarlo.features.deeptica.core.module import construct_deeptica_core
            from pmarlo.features.deeptica._full import DeepTICAConfig
            
            cfg = DeepTICAConfig(**config_data)
            core_net = construct_deeptica_core(cfg, scaler)
            
            # Try to load the state dict
            try:
                # Check if keys have 'inner.' prefix
                if any(k.startswith("inner.") for k in actual_state.keys()):
                    # Remove 'inner.' prefix
                    cleaned_state = {}
                    for key, value in actual_state.items():
                        if key.startswith("inner."):
                            cleaned_state[key[6:]] = value
                        else:
                            cleaned_state[key] = value
                    core_net.load_state_dict(cleaned_state, strict=False)
                else:
                    core_net.load_state_dict(actual_state, strict=False)
                
                core_net.eval()
                network = core_net
                logger.info("Successfully loaded network using alternative method")
            except Exception as net_error:
                logger.error(f"Failed to load network: {net_error}")
                raise
            
            history = {}
        
        # Determine output directory
        if output_dir is None:
            output_dir = model_base_path.parent
        else:
            output_dir = Path(output_dir)
        
        # Load feature specification
        spec_path = Path(__file__).parent / "app" / "feature_spec.yaml"
        if not spec_path.exists():
            raise FileNotFoundError(
                f"Feature specification not found at {spec_path}. "
                "Make sure you're running this from the pmarlo_webapp directory."
            )
        
        logger.info(f"Loading feature specification from {spec_path}")
        with spec_path.open("r", encoding="utf-8") as spec_file:
            feature_spec = yaml.safe_load(spec_file)

        normalized_spec = canonicalize_feature_spec(feature_spec)
        expected_features = int(normalized_spec.n_features)
        scaler_mean = np.asarray(getattr(scaler, "mean_", []))
        actual_features = int(scaler_mean.shape[0]) if scaler_mean.size else 0
        if expected_features <= 0:
            raise RuntimeError(
                "feature_spec.yaml does not define any molecular features. "
                "Add feature definitions before exporting a CV bias bundle."
            )
        if actual_features != expected_features:
            raise RuntimeError(
                "Feature count mismatch detected. "
                f"feature_spec.yaml defines {expected_features} feature(s) but the "
                f"trained model scaler expects {actual_features}. "
                "Ensure the model was trained on shards created with the same feature specification."
            )
        
        logger.info("Exporting CV model with bias potential for OpenMM integration...")
        logger.info(f"  Output directory: {output_dir}")
        logger.info(f"  Bias strength: 10.0 kJ/mol")
        
        cv_bundle = export_cv_model(
            network=network,
            scaler=scaler,
            history=history,
            output_dir=output_dir,
            model_name="deeptica_cv_model",
            bias_strength=10.0,
            feature_spec=feature_spec,
        )
        
        logger.info("=" * 60)
        logger.info("CV bias bundle exported successfully!")
        logger.info("=" * 60)
        logger.info(f"Model path:    {cv_bundle.model_path}")
        logger.info(f"Scaler path:   {cv_bundle.scaler_path}")
        logger.info(f"Config path:   {cv_bundle.config_path}")
        logger.info(f"Metadata path: {cv_bundle.metadata_path}")
        logger.info(f"CV dimensions: {cv_bundle.cv_dim}")
        logger.info(f"Feature hash:  {cv_bundle.feature_spec_hash}")
        logger.info("=" * 60)
        logger.info("")
        logger.info("You can now use this model for metabiased simulations in:")
        logger.info("  - pmarlo_webapp (Sampling tab)")
        logger.info("  - Example programs that use CV-informed sampling")
        
    except Exception as e:
        logger.error(f"Failed to export CV bundle: {e}", exc_info=True)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Export CV bias bundle from a trained DeepTICA model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export CV bundle for a specific model (output to same directory)
  python export_cv_bundle.py app_output/models/deeptica-20251108-175716
  
  # Export CV bundle to a specific output directory
  python export_cv_bundle.py app_output/models/deeptica-20251108-175716 -o /path/to/output
  
  # Export CV bundle for the model in training directory
  python export_cv_bundle.py app_output/models/training-20251108-173316/deeptica-20251108-175716
        """
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the model base (without suffix), e.g., deeptica-20251108-175716"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory for the CV bundle (default: same as model directory)"
    )
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    output_dir = Path(args.output) if args.output else None
    
    export_cv_bundle_from_model(model_path, output_dir)


if __name__ == "__main__":
    main()

