ookimport sys
import json
from pathlib import Path

# Add app to path
sys.path.insert(0, r'C:\Users\konrad_guest\Documents\GitHub\pmarlo\pmarlo_webapp\app')

# Test 1: Check state.json content
state_path = Path(r'C:\Users\konrad_guest\Documents\GitHub\pmarlo\pmarlo_webapp\app_output\state.json')
print(f"State path exists: {state_path.exists()}")
print(f"State path: {state_path}")

if state_path.exists():
    with open(state_path, 'r') as f:
        data = json.load(f)
    print(f"\nDirect JSON read:")
    print(f"  Runs: {len(data.get('runs', []))}")
    print(f"  Shards: {len(data.get('shards', []))}")
    print(f"  Models: {len(data.get('models', []))}")
    print(f"  Builds: {len(data.get('builds', []))}")
    print(f"  Conformations: {len(data.get('conformations', []))}")

# Test 2: Check what WorkspaceLayout resolves to
from backend.layout import WorkspaceLayout
layout = WorkspaceLayout.from_app_package()
print(f"\nWorkspaceLayout:")
print(f"  app_root: {layout.app_root}")
print(f"  workspace_dir: {layout.workspace_dir}")
print(f"  state_path: {layout.state_path}")
print(f"  State exists: {layout.state_path.exists()}")

# Test 3: Load state through PersistentState
from backend.state import PersistentState
state = PersistentState(layout.state_path)
print(f"\nPersistentState:")
print(f"  Runs: {len(state.runs)}")
print(f"  Shards: {len(state.shards)}")
print(f"  Models: {len(state.models)}")
print(f"  Builds: {len(state.builds)}")
print(f"  Conformations: {len(state.conformations)}")

# Test 4: Build context like the app does
from core.context import build_context
ctx = build_context()
print(f"\nContext (as app sees it):")
print(f"  Workspace: {ctx.backend.layout.workspace_dir}")
print(f"  State path: {ctx.backend.state.path}")
summary = ctx.backend.sidebar_summary()
print(f"  Sidebar summary: {summary}")

