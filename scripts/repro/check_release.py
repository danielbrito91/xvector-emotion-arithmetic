"""Confere o conjunto de publicação sem alterar o Git ou executar experimentos."""

from __future__ import annotations

import ast
import json
import re
import sys
import tomllib
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    manifest = Path(__file__).with_name('release_files.txt')
    files = [
        line.strip()
        for line in manifest.read_text().splitlines()
        if line.strip() and not line.startswith('#')
    ]
    errors = []
    if len(files) != len(set(files)):
        errors.append('Arquivos duplicados no manifesto.')
    for rel in files:
        path = root / rel
        if Path(rel).is_absolute() or '..' in Path(rel).parts:
            errors.append(f'Caminho fora do repositório: {rel}')
            continue
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            errors.append(f'Arquivo ausente: {rel}')
            continue
        if rel.startswith('data/') and not rel.startswith('data/tau/'):
            errors.append(f'Dado fora do escopo: {rel}')
        if path.suffix == '.py':
            tree = ast.parse(path.read_text(), filename=rel)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                    if (root / node.module.replace('.', '/')).is_dir():
                        modules += [f'{node.module}.{alias.name}' for alias in node.names]
                else:
                    continue
                for name in modules:
                    module = name.replace('.', '/')
                    candidates = [f'{module}.py', f'{module}/__init__.py']
                    namespace = (root / module).is_dir()
                    if name.startswith(('src.', 'scripts.')) and not namespace:
                        if not any(candidate in files for candidate in candidates):
                            errors.append(f'Import local ausente: {rel}: {name}')
                    sibling = path.parent / f'{module}.py'
                    if sibling.is_file() and str(sibling.relative_to(root)) not in files:
                        errors.append(f'Import vizinho fora do manifesto: {rel}: {name}')
        elif path.suffix == '.json':
            json.loads(path.read_text())
        elif path.suffix == '.toml' or path.name == 'uv.lock':
            tomllib.loads(path.read_text())
    for rel in ['README.md', 'scripts/repro/README.md']:
        for target in re.findall(r'\]\(([^)]+)\)', (root / rel).read_text()):
            if '://' not in target and not target.startswith('#'):
                resolved = ((root / rel).parent / target).resolve()
                if (
                    not resolved.is_relative_to(root)
                    or str(resolved.relative_to(root)) not in files
                ):
                    errors.append(f'Link local fora do manifesto em {rel}: {target}')
    print(json.dumps({'files': len(files), 'errors': errors}, indent=2, ensure_ascii=False))
    return bool(errors)


if __name__ == '__main__':
    sys.exit(main())
