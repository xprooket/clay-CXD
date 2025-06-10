# ⚠️ Duplicate Models Directory Notice

## Issue Identified

There are currently duplicate `models/` directories:
- `clay/models/`
- `cxd-classifier/models/`

Both contain identical files (Asym.py, BoW.py, CNN.py, etc.). This could be confusing for users and contributors.

## Recommendation

Consider one of these approaches:

1. **Consolidate**: Move models to a shared location like `shared/models/` and import from both projects
2. **Clarify**: Add README files in each explaining the relationship
3. **Remove**: Keep models only in the main CXD project since Clay imports from CXD

## Not Blocking Release

This doesn't prevent the repository from working, but should be addressed in a future cleanup.

---
*This file can be removed after the issue is resolved.*
