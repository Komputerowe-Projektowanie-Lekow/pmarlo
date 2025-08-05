# test_local.py
try:
    # Test imports - using the installed package name
    from pmarlo import Pipeline, Protein, Simulation, ReplicaExchange, MarkovStateModel
    from pmarlo.manager.checkpoint_manager import CheckpointManager
    
    print("✅ All imports successful!")
    
    # Test basic functionality
    # Add any basic instantiation tests here
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")