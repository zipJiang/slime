"""Verify the collector uses the recorded services and GPU topology."""
import hashlib
from pathlib import Path
import subprocess
from urllib.parse import urlparse


JUDGE_CHECKPOINT=Path('/weka/projects/bvandur1/zjiang31/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654')


def validate_infrastructure(args,infrastructure):
    from pilot_topology import required_gpus
    if infrastructure.get('schema')!='browsecomp-zero-warmup-pilot-infrastructure-v1':
        raise ValueError('Unknown pilot infrastructure contract')
    retriever=args.retriever_code.resolve();index=(retriever/'indexes/qwen3-embedding-0.6b').resolve()
    commit=subprocess.check_output(['git','-C',str(retriever),'rev-parse','HEAD'],text=True).strip()
    clean=not bool(subprocess.check_output(
        ['git','-C',str(retriever),'status','--porcelain'],text=True).strip())
    hashes={}
    for path in sorted(index.glob('*')):
        if path.is_file():
            with path.open('rb') as stream:
                hashes[path.name]=hashlib.file_digest(stream,'sha256').hexdigest()
    endpoints={urlparse(args.retriever_url).hostname,urlparse(args.judge_url).hostname}
    gpus=infrastructure.get('gpus',{});required=infrastructure.get('required_gpus',{})
    training_gpus=required.get('train',0)+required.get('train_worker',0)
    if (infrastructure.get('retriever_commit')!=commit
            or infrastructure.get('retriever_tree_clean') is not True or not clean
            or infrastructure.get('retriever_client_sha256')!=hashlib.sha256(
                (retriever/'retriever/serve/client.py').read_bytes()).hexdigest()
            or Path(infrastructure.get('retriever_index','')).resolve()!=index
            or infrastructure.get('retriever_index_sha256')!=hashes
            or Path(infrastructure.get('judge_checkpoint','')).resolve()!=JUDGE_CHECKPOINT.resolve()
            or endpoints!={infrastructure.get('ips',{}).get('aux')}
            or required!=required_gpus(infrastructure.get('jobs',{}),train_gpus=training_gpus)
            or any(int(gpus.get(role,0))<count for role,count in required.items())):
        raise ValueError('Pilot infrastructure differs from the recorded services or resources')

