import hashlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pilot_infrastructure import JUDGE_CHECKPOINT, validate_infrastructure


def infrastructure(tmp_path, training):
    retriever=tmp_path/'retriever'
    client=retriever/'retriever/serve/client.py'
    client.parent.mkdir(parents=True);client.write_text('# client\n')
    index=retriever/'indexes/qwen3-embedding-0.6b'
    index.mkdir(parents=True);(index/'index.bin').write_bytes(b'index')
    required=dict(training,inference=2,replica=1,aux=2)
    manifest=dict(schema='browsecomp-zero-warmup-pilot-infrastructure-v1',
        retriever_commit='revision',retriever_tree_clean=True,
        retriever_client_sha256=hashlib.sha256(client.read_bytes()).hexdigest(),
        retriever_index=str(index),
        retriever_index_sha256={'index.bin':hashlib.sha256(b'index').hexdigest()},
        judge_checkpoint=str(JUDGE_CHECKPOINT),ips={'aux':'10.0.0.1'},
        jobs={role:i for i,role in enumerate(required,1)},
        required_gpus=required,gpus=required.copy())
    args=SimpleNamespace(retriever_code=retriever,
        retriever_url='http://10.0.0.1:8125',judge_url='http://10.0.0.1:8131/v1')
    return args,manifest


@pytest.mark.parametrize('training',[{'train':2},{'train':4},{'train':2,'train_worker':2}])
def test_collector_accepts_supported_trainer_layouts(tmp_path,training):
    args,manifest=infrastructure(tmp_path,training)
    with patch('pilot_infrastructure.subprocess.check_output',side_effect=['revision','']):
        validate_infrastructure(args,manifest)


def test_collector_rejects_insufficient_allocated_gpus(tmp_path):
    args,manifest=infrastructure(tmp_path,{'train':2})
    manifest['gpus']['train']=1
    with patch('pilot_infrastructure.subprocess.check_output',side_effect=['revision','']), \
         pytest.raises(ValueError,match='resources'):
        validate_infrastructure(args,manifest)


@pytest.mark.parametrize('training',[{'train':3},{'train':1,'train_worker':1}])
def test_collector_rejects_unsupported_trainer_layouts(tmp_path,training):
    args,manifest=infrastructure(tmp_path,training)
    with patch('pilot_infrastructure.subprocess.check_output',side_effect=['revision','']), \
         pytest.raises(ValueError):
        validate_infrastructure(args,manifest)
