#!/usr/bin/env bash
# Assemble the GitHub Pages site: the page in docs/ plus the package sources the demo runs in Pyodide.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf _site
cp -R docs _site
mkdir -p _site/py/examples
cp -R evolvemas _site/py/
rm -rf _site/py/evolvemas/__pycache__
cp examples/debate.py _site/py/examples/
mkdir -p _site/py/examples/data
cp examples/data/*.jsonl _site/py/examples/data/
python3 - <<'PY'
import json, pathlib, tomllib
root = pathlib.Path("_site/py")
files = sorted(str(p.relative_to(root)) for p in list(root.rglob("*.py")) + list(root.rglob("*.jsonl")))
version = tomllib.loads(pathlib.Path("pyproject.toml").read_text())["project"]["version"]
(root / "manifest.json").write_text(json.dumps({"version": version, "files": files}, indent=1))
runs = pathlib.Path("_site/runs")
ids = sorted(p.stem for p in runs.glob("*.json") if p.stem != "index") if runs.exists() else []
if runs.exists():
    (runs / "index.json").write_text(json.dumps({"traces": ids}))
print("site built:", len(files), "python files, version", version, "| traces:", ", ".join(ids) or "none")
PY
