"""Audit candidate Git content, including staged blobs when --staged is used."""
import argparse
from pathlib import Path, PurePosixPath
import re
import subprocess

ROOT_FILES = {'.gitignore', 'README.md', 'pyproject.toml'}
SUFFIXES = {'src': {'.py'}, 'scripts': {'.py'}, 'tools': {'.py'}, 'configs': {'.yaml'}, 'docs': {'.md', '.json'}}
PATTERNS = [rb'gh[pousr]_[A-Za-z0-9]{30,}', rb'github_pat_[A-Za-z0-9_]{30,}', rb'AKIA[0-9A-Z]{16}', rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----', rb'https?://[^\s/@:]+:[^\s/@]+@', rb'/(?:Users|home)/[A-Za-z0-9_.-]+/']

def inspect(name, content, mode='100644'):
    path = PurePosixPath(name)
    allowed = name in ROOT_FILES or name == '.githooks/pre-commit' or (len(path.parts) > 1 and path.parts[0] in SUFFIXES and path.suffix in SUFFIXES[path.parts[0]])
    errors = []
    if not allowed: errors.append('not in source/config/docs allowlist')
    if mode not in ('100644', '100755'): errors.append('symlinks/submodules prohibited')
    if len(content) > 1024 * 1024: errors.append('file exceeds 1 MiB')
    try: content.decode('utf-8')
    except UnicodeDecodeError: errors.append('not UTF-8 text')
    if b'\x00' in content: errors.append('binary NUL')
    if any(re.search(pattern, content) for pattern in PATTERNS): errors.append('credential or personal path pattern')
    return errors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--staged', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    def git(*argv): return subprocess.check_output(['git', '-C', str(root), *argv])
    entries = []
    if args.staged:
        for row in git('ls-files', '--stage', '-z').split(b'\x00'):
            if not row: continue
            metadata, name = row.split(b'\t', 1)
            mode, sha, stage = metadata.decode().split()
            if stage != '0': raise RuntimeError('Unmerged index')
            entries.append((name.decode(), git('cat-file', 'blob', sha), mode))
    else:
        paths = git('ls-files', '--cached', '--others', '--exclude-standard', '-z').split(b'\x00')
        for name in sorted(set(paths)):
            if not name: continue
            path = root / name.decode()
            if not path.exists(): continue
            entries.append((name.decode(), path.read_bytes(), '120000' if path.is_symlink() else '100644'))
    failures = [(name, inspect(name, data, mode)) for name, data, mode in entries]
    failures = [(name, errors) for name, errors in failures if errors]
    for name, errors in failures: print(name + ': ' + ', '.join(errors))
    if failures: raise SystemExit(1)
    print(f'PASS: {len(entries)} text files, {sum(len(data) for _, data, _ in entries):,} bytes; max 1 MiB/file')

if __name__ == '__main__': main()
