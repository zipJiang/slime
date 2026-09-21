"""The production profile resolves only the frozen robust harness."""

import json
import os
from pathlib import Path
import subprocess
import sys


def test_robust_runtime_snapshot_and_contract():
    scripts = Path(__file__).resolve().parent
    code = """
import json
import runtime_active
runtime_active.verify_harness()
import ppo_runtime
import examples.locbench.env as env
import step_controller.harness.compaction.summary as summary
contract=ppo_runtime.environment('robust-null-v3')
assert contract['actor_workspace']==contract['compactor_workspace']=='null'
assert contract['compaction_memory_role']=='assistant'
assert contract['compaction_failure']=='terminal_zero'
assert contract['harness_commit']=='be3d85ace36f10f3b4ae71804ec2c3929ed7ce68'
assert str(runtime_active.HARNESS) in env.__file__
assert str(runtime_active.HARNESS) in summary.__file__
print(json.dumps(contract,sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=scripts,
        env=dict(os.environ, LOC_COLLECTION_PROFILE="robust-null-v3"),
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["profile"] == "locbench-robust-null-v3"


def test_activation_replaces_an_already_imported_legacy_harness():
    scripts = Path(__file__).resolve().parent
    code = r'''
import json
from pathlib import Path
import sys

scripts = Path.cwd()
legacy = scripts.parent / "snapshots/harness-v1"
sys.path.insert(0, str(legacy))
from examples.locbench.env import RunConfig as LegacyRunConfig
assert "robust_compaction" not in LegacyRunConfig.__dataclass_fields__

from runtime_v3 import HARNESS, activate
activate()
from examples.locbench.env import RunConfig as RobustRunConfig
import examples.locbench.env as env
print(json.dumps({
    "has_robust_compaction": "robust_compaction" in RobustRunConfig.__dataclass_fields__,
    "source": str(Path(env.__file__).resolve()),
    "harness": str(HARNESS.resolve()),
}))
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=scripts,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["has_robust_compaction"] is True
    assert Path(payload["source"]).is_relative_to(Path(payload["harness"]))
