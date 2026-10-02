"""CPU-only syntax, archived-source integrity and configuration validation."""
import ast
import hashlib
import json
from pathlib import Path
import sys
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'src'))
from aerial_openvla.core.config import TrainConfig, EvalConfig, load_yaml_dataclass
manifest = json.loads((root / 'docs/source_manifest.json').read_text())
for entry in manifest['files']:
    content = (root / entry['path']).read_bytes()
    assert hashlib.sha256(content).hexdigest() == entry['sha256'], entry['path']
    assert len(content) == entry['bytes'], entry['path']
files = list(root.rglob('*.py'))
for path in files:
    ast.parse(path.read_text(), filename=str(path))
configs = list((root / 'configs').rglob('*.yaml'))
for path in configs:
    cfg = load_yaml_dataclass(path, EvalConfig if path.stem.startswith('eval_') else TrainConfig)
    assert cfg.model_type == 'airminivla', path
print(f'PASS: {len(manifest["files"])} source hashes, {len(files)} Python files, {len(configs)} configuration schemas')
