"""Snapshot committed sources without modifying either checkout."""
import hashlib
import json
import subprocess
from pathlib import Path


def snapshot(source, destination):
    if destination.exists():
        raise ValueError(f'Refusing to overwrite source snapshot: {destination}')
    if subprocess.check_output(['git', '-C', str(source), 'diff', 'HEAD', '--name-only'], text=True).strip():
        raise ValueError(f'Commit source changes before snapshotting: {source}')
    names = subprocess.check_output(
        ['git', '-C', str(source), 'ls-files', '-z', '--cached']
    ).decode().split('\0')
    files = {}
    destination.mkdir(parents=True)
    for name in sorted(set(names)):
        if not name:
            continue
        path = source / name
        # Do not follow workspace links into another checkout or storage area.
        if path.is_symlink() or not path.is_file():
            continue
        data = path.read_bytes()
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        target.chmod(path.stat().st_mode & 0o777)
        files[name] = hashlib.sha256(data).hexdigest()
    manifest = dict(source=str(source.resolve()),
                    commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
                    files=files)
    (destination / 'source-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return len(files)


if __name__ == '__main__':
    experiment = Path(__file__).resolve().parents[1]
    root = experiment.parents[2]
    for name, source in [('harness', root/'rollout-controller'), ('slime', experiment.parents[1])]:
        print(name, snapshot(source, experiment/'snapshots'/name))
