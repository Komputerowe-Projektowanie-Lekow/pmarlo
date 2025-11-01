import sys
import os
import json

os.chdir(r'C:\Users\konrad_guest\Documents\GitHub\pmarlo')
sys.path.insert(0, r'pmarlo_webapp\app')

from backend.layout import WorkspaceLayout

layout = WorkspaceLayout.from_app_package()

output = []
output.append(f"State path from layout: {layout.state_path}")
output.append(f"State exists: {layout.state_path.exists()}")
output.append(f"Workspace dir: {layout.workspace_dir}")

# Check the actual state.json
actual_state = r'C:\Users\konrad_guest\Documents\GitHub\pmarlo\pmarlo_webapp\app_output\state.json'
with open(actual_state, 'r') as f:
    data = json.load(f)

output.append(f"\nActual state.json has:")
output.append(f"  Runs: {len(data.get('runs', []))}")
output.append(f"  Shards: {len(data.get('shards', []))}")
output.append(f"  Models: {len(data.get('models', []))}")
output.append(f"  Builds: {len(data.get('builds', []))}")

output.append(f"\nPaths match: {str(layout.state_path).lower() == actual_state.lower()}")
output.append(f"Layout path: {str(layout.state_path)}")
output.append(f"Actual path: {actual_state}")

# Now load via Backend
from backend import Backend
backend = Backend(layout)
output.append(f"\nBackend state has:")
output.append(f"  Runs: {len(backend.state.runs)}")
output.append(f"  Shards: {len(backend.state.shards)}")
output.append(f"  Models: {len(backend.state.models)}")
output.append(f"  Builds: {len(backend.state.builds)}")

result = '\n'.join(output)
print(result)

with open('state_debug_output.txt', 'w') as f:
    f.write(result)

