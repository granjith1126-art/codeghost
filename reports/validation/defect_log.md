# CodeGhost Milestone 8 — Defect Log

| Defect ID | Severity | Description | Root Cause | Fix Applied | Regression Test |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **DEF-01** | High | Windows charmap `UnicodeEncodeError` on terminal printing | `✓` checkmark in Rich console on cp1252 Windows shells | Added `sys.stdout.reconfigure(encoding="utf-8")` and ASCII-safe markers | `tests/test_e2e_full_pipeline.py` |
| **DEF-02** | Medium | `PytestCollectionWarning` on `TestSandbox` | Pytest collects classes named `Test*` by default | Added `__test__ = False` to `TestSandbox` | `tests/test_autofix.py` |
| **DEF-03** | Medium | Regex cast matcher failed on complex arguments | `int\(([a-zA-Z0-9_]+)\)` only matched simple symbols | Changed to `int\(([^)]+)\)` to support expressions | `tests/test_autofix.py::test_generate_defensive_and_exception_fixes` |
| **DEF-04** | Medium | String enum `value` attribute access error | Pydantic `use_enum_values=True` stringifies enums | Used `getattr(obj, "value", str(obj))` | `tests/test_pr_creator.py` |
| **DEF-05** | Low | `--dry-run` flag parsing error | Missing explicit `--dry-run` destination | Added `parser.add_argument("--dry-run")` | `tests/test_e2e_full_pipeline.py::test_cli_mode_scan_execution` |
