# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""
PMARLO: Protein Markov State Model Analysis with Replica Exchange

Main entry point demonstrating both the legacy interface and the new clean API.
"""

# New clean API imports
try:
    # Try package imports first (when installed)
    from pmarlo import LegacyPipeline, Pipeline, Protein
    from pmarlo.manager.checkpoint_manager import CheckpointManager, list_runs
    from pmarlo.pipeline import run_pmarlo
    from pmarlo.simulation.simulation import (
        build_transition_model,
        feature_extraction,
        plot_DG,
        prepare_system,
        production_run,
        relative_energies,
    )
except ImportError:
    # Fall back to relative imports (when running from source)
    from . import (
        Protein,
        Pipeline,
        LegacyPipeline,
    )
    from .pipeline import run_pmarlo
    from .simulation.simulation import (
        prepare_system,
        production_run,
        feature_extraction,
        build_transition_model,
        relative_energies,
        plot_DG,
    )
    from .manager.checkpoint_manager import CheckpointManager, list_runs

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, cast

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
TESTS_DIR = BASE_DIR / "tests" / "data"


def original_pipeline_with_dg():
    pdb_file = TESTS_DIR / "3gd8.pdb"
    dcd_path = TESTS_DIR / "traj.dcd"
    pdb_fixed_path = TESTS_DIR / "3gd8-fixed.pdb"

    try:
        # Initialize and prepare the protein
        print("Initializing Protein...")
        protein = Protein(str(pdb_file), ph=8.0)
        print("Protein initialized successfully.")

        # Save the prepared protein structure
        print("Saving prepared protein structure...")
        protein.save(str(pdb_fixed_path))
        print(f"Prepared protein saved to: {pdb_fixed_path}")

        # Get protein properties
        properties = protein.get_properties()
        print(
            f"Protein prepared: {properties['num_atoms']} atoms, "
            f"{properties['num_residues']} residues"
        )

        # Prepare system and metadynamics
        simulation, meta = prepare_system(
            str(pdb_fixed_path)
        )  # Ensure absolute path is passed

        # Run production
        production_run(steps=None, simulation=simulation, meta=meta)

        # Feature extraction
        states = feature_extraction(
            str(dcd_path), str(pdb_fixed_path)
        )  # Ensure absolute paths are passed

        # Build Markov model and print free energies
        DG = build_transition_model(states)
        print("Free energies (kcal/mol):", DG)
        plot_DG(DG)

        DG = relative_energies(DG)
        print("Relative energies (kcal/mol):", DG)
        plot_DG(DG)

    except Exception as e:
        print(f"An error occurred: {e}")


def test_protein():
    """Test protein preparation functionality."""
    pdb_file = TESTS_DIR / "3gd8.pdb"
    pdb_fixed_path = TESTS_DIR / "3gd8-fixed.pdb"

    try:
        # Initialize and prepare the protein
        print("Testing Protein class...")
        protein = Protein(str(pdb_file), ph=7.0)
        print("Protein initialized successfully.")

        # Save the prepared protein structure
        print("Saving prepared protein structure...")
        protein.save(str(pdb_fixed_path))
        print(f"Prepared protein saved to: {pdb_fixed_path}")

        # Get and print protein properties
        print("Retrieving protein properties...")
        properties = protein.get_properties()
        print("Protein properties:")
        for key, value in properties.items():
            print(f"  {key}: {value}")

        print("✓ Protein test completed successfully.")
    except Exception as e:
        print(f"✗ An error occurred during the test: {e}")


def run_remd_pipeline(run_id=None, continue_run=False, steps=1000, n_states=50):
    """Run the legacy REMD + Enhanced MSM pipeline with checkpoint support."""

    # Use the new LegacyPipeline class instead of the old function
    output_base_dir = Path(__file__).parent.parent / "output"
    pdb_file = TESTS_DIR / "3gd8.pdb"

    legacy_pipeline = LegacyPipeline(
        pdb_file=str(pdb_file),
        output_dir=str(output_base_dir),
        run_id=run_id,
        continue_run=continue_run,
    )

    return legacy_pipeline.run_legacy_remd_pipeline(steps=steps, n_states=n_states)


def run_comparison_analysis():
    """Run both pipelines and compare results."""
    print("=" * 60)
    print("PIPELINE COMPARISON")
    print("=" * 60)

    try:
        # Run original pipeline
        print("\n>>> Running Original Single-Temperature Pipeline...")
        original_pipeline_with_dg()

        # Run new REMD pipeline
        print("\n>>> Running New REMD + Enhanced MSM Pipeline...")
        msm = run_remd_pipeline()

        if msm is not None:
            print("\n>>> Comparison Complete!")
            print("Check output directories for detailed results:")
            print(f"  - Original: {TESTS_DIR}")
            print(f"  - REMD: {BASE_DIR / 'output' / 'replica_exchange'}")
            print(f"  - Enhanced MSM: {BASE_DIR / 'output' / 'msm_analysis'}")

    except Exception as e:
        print(f"Error in comparison analysis: {e}")


def demo_simple_api():
    """Demonstrate the new simple 5-line API with checkpoint support."""
    print("=" * 60)
    print("PMARLO SIMPLE API DEMONSTRATION")
    print("=" * 60)

    # The requested 5-line usage pattern
    print("Five-line usage (as requested):")
    print(
        """
protein = Protein("tests/data/3gd8.pdb", ph=7.0)
replica_exchange = ReplicaExchange(
    "tests/data/3gd8-fixed.pdb",
    temperatures=[300, 310, 320],
)
simulation = Simulation(
    "tests/data/3gd8-fixed.pdb",
    temperature=300,
    steps=1000,
)
markov_state_model = MarkovStateModel()
pipeline = Pipeline("tests/data/3gd8.pdb").run()  # Orchestrates everything
    """
    )

    print("\nNew checkpoint-enabled usage:")
    print(
        """
# Basic usage with auto-continue
results = run_pmarlo('protein.pdb', auto_continue=True)

# Programmatic checkpoint handling
pipeline = Pipeline('protein.pdb', auto_continue=True)
if pipeline.can_continue():
    print(f"Continuing from checkpoint: {pipeline.get_checkpoint_status()}")
results = pipeline.run()

# Specific checkpoint ID
results = run_pmarlo('protein.pdb', checkpoint_id='12345')
    """
    )

    print("\nUltra-simple one-liner with checkpoints:")
    print(
        "results = run_pmarlo("
        "'protein.pdb',"
        "\n"
        "    temperatures=[300, 310, 320],\n"
        "    steps=1000,\n"
        "    auto_continue=True,\n"
        ")"
    )


def demo_checkpoint_api():
    """Demonstrate the checkpoint API for library usage."""
    try:
        print("\n🔍 Checking for interrupted runs...")
        status = check_interrupted_runs()

        if status["has_interrupted"]:
            print(f"Found interrupted run: {status['run_id']}")
            print(f"Status: {status['status']}")
        else:
            print("No interrupted runs found")

        print("\n📝 Available helper functions for library usage:")
        print("  - check_interrupted_runs() -> Dict")
        print("  - continue_run_programmatically(run_id, pdb_file) -> Dict")
        print(
            "  - CheckpointManager.auto_detect_interrupted_run() -> CheckpointManager"
        )

    except Exception as e:
        print(f"Checkpoint API demo failed: {e}")


def run_simple_example():
    """Run a simple example with the new API (minimal steps for testing)."""
    try:
        print("Running simple PMARLO example with checkpoint support...")

        # Use minimal parameters for quick testing
        pdb_file = str(TESTS_DIR / "3gd8.pdb")

        # Create a pipeline with minimal settings and auto-continue
        pipeline = Pipeline(
            pdb_file=pdb_file,
            steps=100,  # Very short for demo
            n_states=10,  # Fewer states for demo
            use_replica_exchange=False,  # Simpler for demo
            output_dir=str(BASE_DIR / "output" / "demo"),
            auto_continue=True,  # Enable auto-continue
        )

        # Check if we can continue from previous run
        if pipeline.can_continue():
            status = pipeline.get_checkpoint_status()
            print(
                "📁 Found existing run - Progress: "
                f"{status['progress']} "
                f"({status['progress_percent']:.1f}%)"
            )
            print(f"🔄 Auto-continuing from: {status['current_stage']}")
        else:
            print("🆕 Starting new run")

        # Show checkpoint manager info
        if pipeline.checkpoint_manager:
            print(f"💾 Checkpoint ID: {pipeline.checkpoint_manager.run_id}")
            print(f"📂 Output directory: {pipeline.checkpoint_manager.run_dir}")

        # This would be the complete run
        # results = pipeline.run()

        # For now, just show the setup
        protein = pipeline.setup_protein()
        print(
            f"✓ Protein setup complete: {protein.get_properties()['num_atoms']} atoms"
        )

        simulation = pipeline.setup_simulation()
        print(f"✓ Simulation setup complete for {simulation.steps} steps")

        pipeline.setup_markov_state_model()
        print(f"✓ MSM setup complete for {pipeline.n_states} states")

        print("\n📋 Current checkpoint status:")
        if pipeline.checkpoint_manager:
            pipeline.checkpoint_manager.print_status()

        # Demo the checkpoint API
        demo_checkpoint_api()

        print("✓ Demo setup complete! To run full simulation, call pipeline.run()")

    except Exception as e:
        print(f"Demo failed (this is expected if test files are missing): {e}")


def demo_new_vs_legacy():
    """Demonstrate the difference between new and legacy APIs."""
    print("=" * 60)
    print("NEW API vs LEGACY API COMPARISON")
    print("=" * 60)

    print("\n🆕 NEW API (Recommended - Simple & Clean with built-in checkpoints):")
    print("=" * 40)
    print(
        """
# Ultra-simple one-liner with auto-resume
from pmarlo import run_pmarlo
results = run_pmarlo(
    "protein.pdb",
    temperatures=[300, 310, 320],
    steps=1000,
    auto_continue=True,
)

# Programmatic checkpoint handling
from pmarlo import Pipeline
pipeline = Pipeline("protein.pdb", auto_continue=True)
if pipeline.can_continue():
    status = pipeline.get_checkpoint_status()
    print(f"Resuming from {status['current_stage']} (" f"{status['progress']})")
results = pipeline.run()

# Specific checkpoint control
pipeline = Pipeline("protein.pdb", checkpoint_id="12345", auto_continue=False)
results = pipeline.run()

# Manual checkpoint detection
from pmarlo.manager import CheckpointManager
interrupted = CheckpointManager.auto_detect_interrupted_run()
if interrupted:
    results = run_pmarlo("protein.pdb", checkpoint_id=interrupted.run_id)
    """
    )

    print("\n🕐 LEGACY API (CLI-focused, manual checkpoint management):")
    print("=" * 40)
    print(
        """
# Legacy checkpoint-enabled pipeline
from pmarlo import LegacyPipeline
legacy = LegacyPipeline("protein.pdb", run_id="12345", continue_run=True)
results = legacy.run_legacy_remd_pipeline(steps=1000, n_states=50)

# CLI usage
python main.py --mode remd --id 12345 --continue
    """
    )

    print("\n💡 NEW API ADVANTAGES:")
    print("   ✅ Automatic checkpoint detection and resume")
    print("   ✅ Library-friendly (no print statements, returns status)")
    print("   ✅ Works with all pipeline types (not just REMD)")
    print("   ✅ Programmatic checkpoint control")
    print("   ✅ Consistent API across all features")
    print("   ✅ Better error handling and recovery")


def _build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for PMARLO."""
    parser = argparse.ArgumentParser(
        description=(
            "PMARLO: Protein Markov State Model Analysis with Replica Exchange"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            """
Examples:
  python main.py --mode original          # Run original single-T pipeline
  python main.py --mode remd             # Run new REMD + enhanced MSM pipeline
  python main.py --mode compare          # Run both pipelines for comparison
  python main.py --mode test             # Test protein preparation only
  python main.py --mode simple           # Demonstrate new simple API
  python main.py --mode demo             # Run simple demo with new API
  python main.py --mode comparison       # Compare new vs legacy APIs
  python main.py --mode checkpoint-demo  # Demonstrate checkpoint functionality
            """
        ),
    )
    _add_mode_argument(parser)
    _add_steps_argument(parser)
    _add_states_argument(parser)
    _add_random_state_argument(parser)
    _add_id_argument(parser)
    _add_continue_argument(parser)
    _add_no_auto_continue_argument(parser)
    _add_list_runs_argument(parser)
    return parser


def _add_mode_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mode",
        choices=[
            "original",
            "remd",
            "compare",
            "test",
            "simple",
            "demo",
            "comparison",
            "checkpoint-demo",
        ],
        default="simple",
        help="Analysis mode to run (default: simple)",
    )


def _add_steps_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of simulation steps (default: 1000 for fast testing)",
    )


def _add_states_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--states",
        type=int,
        default=50,
        help="Number of MSM states (default: 50)",
    )


def _add_random_state_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed for random number generators",
    )


def _add_id_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--id", type=str, help="5-digit run ID for checkpoint management"
    )


def _add_continue_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--continue",
        dest="continue_run",
        action="store_true",
        help="Continue from last successful checkpoint (legacy mode only)",
    )


def _add_no_auto_continue_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-auto-continue",
        action="store_true",
        help="Disable automatic continuation of interrupted runs",
    )


def _add_list_runs_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List all available runs and their status",
    )


def _parse_args_or_default(parser: argparse.ArgumentParser):
    """Parse CLI args, defaulting to remd mode if none provided."""
    if len(sys.argv) == 1:
        return parser.parse_args(["--mode", "remd"])
    return parser.parse_args()


def _handle_list_runs_flag(args) -> bool:
    """Handle the --list-runs flag; return True if handled and program should exit."""
    if getattr(args, "list_runs", False):
        list_runs()
        return True
    return False


def _print_program_header(mode: str) -> None:
    print("PMARLO - Protein Markov State Model Analysis")
    print("https://github.com/yourusername/pmarlo")
    print(f"Mode: {mode}")
    print()


def _run_mode_test(_args) -> None:
    test_protein()


def _run_mode_original(_args) -> None:
    original_pipeline_with_dg()


def _run_mode_remd(args) -> None:
    run_remd_pipeline(
        run_id=args.id,
        continue_run=args.continue_run,
        steps=args.steps,
        n_states=args.states,
    )


def _run_mode_compare(_args) -> None:
    run_comparison_analysis()


def _run_mode_simple(_args) -> None:
    demo_simple_api()


def _run_mode_demo(_args) -> None:
    run_simple_example()


def _run_mode_checkpoint_demo(_args) -> None:
    demo_checkpoint_functionality()


def _run_mode_comparison(_args) -> None:
    demo_new_vs_legacy()


def _get_mode_handlers():
    return {
        "test": _run_mode_test,
        "original": _run_mode_original,
        "remd": _run_mode_remd,
        "compare": _run_mode_compare,
        "simple": _run_mode_simple,
        "demo": _run_mode_demo,
        "checkpoint-demo": _run_mode_checkpoint_demo,
        "comparison": _run_mode_comparison,
    }


def main():
    """Main entry point with command-line argument parsing and dispatch."""
    parser = _build_argument_parser()
    args = _parse_args_or_default(parser)
    if _handle_list_runs_flag(args):
        return
    random.seed(args.random_state)
    np.random.seed(args.random_state)
    _print_program_header(args.mode)
    handlers = _get_mode_handlers()
    handler = handlers.get(args.mode)
    if handler is None:
        raise ValueError(f"Unknown mode: {args.mode}")
    handler(args)


def demo_checkpoint_functionality():
    """Demonstrate checkpoint functionality for library usage."""
    print("=" * 60)
    print("CHECKPOINT FUNCTIONALITY DEMONSTRATION")
    print("=" * 60)

    # Show different ways to use checkpoints
    print("\n1. 🔍 Auto-detect interrupted runs:")
    print(
        """
from pmarlo.manager import CheckpointManager

# Find interrupted runs
interrupted = CheckpointManager.auto_detect_interrupted_run("output")
if interrupted:
    print(f"Found interrupted run: {interrupted.run_id}")
    status = interrupted.get_run_summary()
    print(f"Progress: {status['progress']} - Stage: {status['current_stage']}")
    """
    )

    print("\n2. 🚀 Auto-continue in pipeline:")
    print(
        """
# Automatically continue if interrupted run exists
results = run_pmarlo('protein.pdb', auto_continue=True)

# Or with explicit pipeline
pipeline = Pipeline('protein.pdb', auto_continue=True)
if pipeline.can_continue():
    print("Continuing previous run...")
results = pipeline.run()
    """
    )

    print("\n3. 🎯 Specific checkpoint control:")
    print(
        """
# Continue specific run
results = run_pmarlo('protein.pdb', checkpoint_id='12345')

# Check status before running
pipeline = Pipeline('protein.pdb', checkpoint_id='12345', auto_continue=False)
status = pipeline.get_checkpoint_status()
if status and status['status'] == 'failed':
    print(f"Previous run failed at: {status['current_stage']}")
    print("Retrying failed step...")
results = pipeline.run()
    """
    )

    print("\n4. 📊 Programmatic status checking:")
    print(
        """
# List all runs
from pmarlo.manager import list_runs
list_runs("output")

# Get detailed status
checkpoint = CheckpointManager.load_existing_run('12345')
summary = checkpoint.get_run_summary()
print(f"Run {summary['run_id']}: {summary['status']} - {summary['progress']}")
    """
    )

    print("\n💡 KEY ADVANTAGES FOR LIBRARY USAGE:")
    print("   • Automatic detection and continuation")
    print("   • No user interaction required")
    print("   • Programmatic status checking")
    print("   • Works seamlessly in scripts and notebooks")
    print("   • Consistent across all pipeline types")


def check_interrupted_runs():
    """Helper function to check for interrupted runs programmatically."""
    interrupted = CheckpointManager.auto_detect_interrupted_run()
    if interrupted:
        return {
            "has_interrupted": True,
            "run_id": interrupted.run_id,
            "status": interrupted.get_run_summary(),
        }
    return {"has_interrupted": False}


def continue_run_programmatically(
    run_id: Optional[str] = None, pdb_file: Optional[str] = None
) -> Dict[str, Any]:
    """Continue a run programmatically without CLI interaction."""
    if run_id:
        # Continue specific run
        try:
            cm = CheckpointManager.load_existing_run(run_id)
            config: Dict[str, Any] = cm.load_config()
            pdb_file = config.get("pdb_file") or pdb_file

            if not pdb_file:
                raise ValueError("PDB file not found in config and not provided")

            return cast(
                Dict[str, Any],
                run_pmarlo(pdb_file, checkpoint_id=run_id, auto_continue=True),
            )
        except FileNotFoundError:
            return {"error": f"Run {run_id} not found"}
    else:
        # Auto-detect and continue
        if not pdb_file:
            raise ValueError("PDB file must be provided for auto-continue")

        return cast(Dict[str, Any], run_pmarlo(pdb_file, auto_continue=True))


if __name__ == "__main__" or Path(__file__).stem == "__main__":
    # This allows the module to be run both as a script and with python -m
    main()
