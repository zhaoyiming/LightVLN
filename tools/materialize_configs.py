"""Render machine-local path placeholders; never modify the archived configs."""
import argparse
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--root', required=True)
p.add_argument('--output', default='local_configs')
a = p.parse_args()
root = Path(a.root).expanduser()
if not root.is_absolute():
    p.error('--root must be absolute')
source = Path(__file__).resolve().parents[1] / 'configs'
output = Path(a.output).resolve()
if output == source or source in output.parents:
    p.error('output must be outside archived configs')
for config in sorted(source.rglob('*.yaml')):
    dest = output / config.relative_to(source)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(dest)
    dest.write_text(config.read_text().replace('/external/lightvln', str(root)))
print(f'Rendered configs to {output}; inspect remaining relative third_party/model paths.')
