"""Create a fresh frozen TRACE adapter over the existing audited harness core."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def prepare(experiment, source):
    experiment, source = Path(experiment), Path(source)
    base = experiment/'snapshots/harness'
    target = experiment/'snapshots/harness-trace-v1'
    if target.exists():
        raise ValueError('TRACE snapshot already exists; refusing to mutate frozen sources')
    stage = target.with_name('.harness-trace-v1-preparing')
    if stage.exists():
        raise ValueError('A previous snapshot preparation must be inspected first')
    manifest = json.loads((base/'source-manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        if hashlib.sha256((base/relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Original snapshot differs: {relative}')
    shutil.copytree(base, stage, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    overlays = {}
    for relative in ['examples/browsercomp_plus/env.py', 'examples/browsercomp_plus/browser.py']:
        shutil.copy2(source/relative, stage/relative)
        overlays[relative] = hashlib.sha256((stage/relative).read_bytes()).hexdigest()
    manifest['files'].update(overlays)
    manifest.update(environment_profile='trace96k', overlay_source=str(source.resolve()),
        overlays=overlays, base_manifest_sha256=hashlib.sha256(
            (base/'source-manifest.json').read_bytes()).hexdigest())
    (stage/'source-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    stage.rename(target)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    print(prepare(Path(__file__).resolve().parents[1], args.source))
