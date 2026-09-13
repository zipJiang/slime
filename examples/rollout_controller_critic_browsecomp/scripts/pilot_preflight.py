"""Pure, CPU-safe validation shared by pilot preparation and execution."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

from pilot_data import select_questions
from ppo_initialization import zero_warmup_role_arguments
from warmstart_candidate import digest, require_pilot_candidate


SCHEMA = 'browsecomp-zero-warmup-preflight-v1'
EXPERIMENT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT = EXPERIMENT/'data/split.json'
DEFAULT_COLLECTION_MANIFEST = EXPERIMENT/'runs/base-v2/collection/manifest.json'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_url(value, name):
    parsed = urlparse(value)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname:
        raise ValueError(f'{name} must be an HTTP(S) URL')
    return value.rstrip('/')


def build(*, candidate_path, context_source, schedule_audit, cases,
          retriever_code, base_actor, updates, batch_size, critic_only_steps,
          train_gpus, rollout_gpus, critic_replica_host, retriever_url,
          judge_url, split_path=DEFAULT_SPLIT,
          collection_manifest=DEFAULT_COLLECTION_MANIFEST):
    candidate_path = Path(candidate_path).resolve()
    context_source = Path(context_source).resolve()
    schedule_audit = Path(schedule_audit).resolve()
    cases = Path(cases).resolve()
    retriever_code = Path(retriever_code).resolve()
    split_path = Path(split_path).resolve()
    collection_manifest = Path(collection_manifest).resolve()
    for path, kind in ((context_source, 'file'), (schedule_audit, 'file'),
                       (cases, 'file'), (retriever_code, 'directory'),
                       (split_path, 'file'), (collection_manifest, 'file')):
        exists = path.is_file() if kind == 'file' else path.is_dir()
        if not exists:
            raise ValueError(f'Pilot {kind} is missing: {path}')
    if (updates, batch_size, critic_only_steps) != (2, 6, 0):
        raise ValueError('Pilot must be exactly two six-question joint updates with zero warmup')
    if (train_gpus, rollout_gpus) != (4, 2):
        raise ValueError('Pilot requires four trainer GPUs and two rollout GPUs')
    if not critic_replica_host:
        raise ValueError('Pilot critic replica host is required')
    schedule = json.loads(schedule_audit.read_text())
    batches = schedule.get('batches')
    if (schedule.get('schema') != 'browsecomp-zero-warmup-pilot-schedule-v1'
            or schedule.get('updates') != updates
            or schedule.get('batch_size') != batch_size
            or not isinstance(batches, list) or len(batches) != updates
            or any(not isinstance(batch, list) or len(batch) != batch_size for batch in batches)):
        raise ValueError('Pilot schedule does not match the fixed two-update design')
    identities = [str(question) for batch in batches for question in batch]
    if len(set(identities)) != len(identities):
        raise ValueError('Pilot schedule repeats a question')
    split=json.loads(split_path.read_text())
    manifest=json.loads(collection_manifest.read_text())
    expected=select_questions(split,manifest,updates=updates,batch_size=batch_size)
    expected['split_sha256']=sha256(split_path)
    for key,value in expected.items():
        if schedule.get(key)!=value:
            raise ValueError(f'Pilot schedule differs from regenerated frozen selection: {key}')
    candidate = require_pilot_candidate(candidate_path)
    args = SimpleNamespace(num_critic_only_steps=critic_only_steps,
        start_rollout_id=None, load=str(Path(base_actor).resolve()), ckpt_step=None)
    _, _, lineage = zero_warmup_role_arguments(args, candidate_path, context_source)
    return dict(schema=SCHEMA, passed=True,
        candidate=str(candidate_path), candidate_sha256=digest(candidate_path),
        candidate_base_actor=str(Path(candidate['base_actor']).resolve()),
        lineage=lineage, context_source=str(context_source),
        context_source_sha256=sha256(context_source),
        schedule=str(schedule_audit), schedule_sha256=digest(schedule_audit),
        schedule_batches=batches, split=str(split_path), split_sha256=sha256(split_path),
        collection_manifest=str(collection_manifest),
        collection_manifest_sha256=sha256(collection_manifest),
        cases=str(cases), cases_sha256=sha256(cases),
        retriever_code=str(retriever_code), updates=updates, batch_size=batch_size,
        critic_only_steps=critic_only_steps, train_gpus=train_gpus,
        rollout_gpus=rollout_gpus, replica_gpus=1,
        auxiliary_gpus=2, total_gpus=9,
        critic_replica_host=critic_replica_host,
        retriever_url=checked_url(retriever_url, 'retriever URL'),
        judge_url=checked_url(judge_url, 'judge URL'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--context-source', type=Path, required=True)
    parser.add_argument('--schedule-audit', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--retriever-code', type=Path, required=True)
    parser.add_argument('--split', type=Path, default=DEFAULT_SPLIT)
    parser.add_argument('--collection-manifest', type=Path, default=DEFAULT_COLLECTION_MANIFEST)
    parser.add_argument('--base-actor', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--critic-only-steps', type=int, default=0)
    parser.add_argument('--train-gpus', type=int, default=4)
    parser.add_argument('--rollout-gpus', type=int, default=2)
    parser.add_argument('--critic-replica-host', required=True)
    parser.add_argument('--retriever-url', required=True)
    parser.add_argument('--judge-url', required=True)
    args = parser.parse_args()
    output = args.run/'preflight.json'
    if output.exists():
        raise ValueError(f'Refusing to replace existing preflight: {output}')
    result = build(candidate_path=args.candidate, context_source=args.context_source,
        schedule_audit=args.schedule_audit, cases=args.cases,
        retriever_code=args.retriever_code, base_actor=args.base_actor,
        split_path=args.split,collection_manifest=args.collection_manifest,
        updates=args.updates, batch_size=args.batch_size,
        critic_only_steps=args.critic_only_steps, train_gpus=args.train_gpus,
        rollout_gpus=args.rollout_gpus, critic_replica_host=args.critic_replica_host,
        retriever_url=args.retriever_url, judge_url=args.judge_url)
    write(output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
