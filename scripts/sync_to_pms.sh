#!/bin/bash
# Sync engine_v9 to PMS — copies files and rewrites imports
# Usage: ./scripts/sync_to_pms.sh

SRC="/var/home/deucebucket/ai-drive/clanker-lang/engine_v9"
DST="/var/home/deucebucket/ai-drive/poor-mans-switch/pms/data/engine_v9"

echo "Syncing engine_v9 → PMS..."
rm -rf "$DST"
cp -r "$SRC" "$DST"

# Rewrite imports (skip tests and __pycache__)
find "$DST" -name "*.py" -not -path "*__pycache__*" -not -path "*/tests/*" \
    -exec sed -i 's/from engine_v9\./from pms.data.engine_v9./g' {} +

# Clean pycache
find "$DST" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

# Verify
python3 -c "
import sys; sys.path.insert(0,'/var/home/deucebucket/ai-drive/poor-mans-switch')
from pms.data.engine_v9.pipeline import compute_vadug
r, t = compute_vadug('test')
print(f'OK: V9 loaded in PMS, V={r.v}')
" || echo "FAILED: V9 import error in PMS"

echo "Done."
