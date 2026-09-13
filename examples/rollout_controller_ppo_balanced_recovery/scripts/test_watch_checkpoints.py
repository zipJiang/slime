"""CPU supervisors must dispatch SIF audits to the allocated execution host."""
from unittest.mock import patch

import pytest

from watch_checkpoints import require_success, validation_command


def test_cpu_supervisor_routes_to_driver_allocation():
    command = ['bash', '/shared/sif.sh', 'python', '/shared/audit.py', '/shared/checkpoint']
    with patch('watch_checkpoints.shutil.which', return_value=None):
        result = validation_command(command, '360795.17')
    assert result == ['srun', '--jobid=360795', '--overlap', '--mem=0',
        '--cpu-bind=none', '-N1', '-n1', '-c4', *command]


def test_gpu_host_uses_local_sif():
    command = ['bash', '/shared/sif.sh', 'python', '/shared/audit.py']
    with patch('watch_checkpoints.shutil.which', return_value='/usr/bin/apptainer'):
        assert validation_command(command, '360795.17') == command


def test_invalid_driver_allocation_is_rejected():
    with patch('watch_checkpoints.shutil.which', return_value=None):
        with pytest.raises(ValueError, match='numeric driver allocation'):
            validation_command(['bash', '/shared/sif.sh'], 'invalid')


def test_failed_validation_cannot_finish_successfully():
    with pytest.raises(RuntimeError, match='iter_0000041'):
        require_success({'iter_0000041': {'passed': False}})
    require_success({'iter_0000041': {'passed': True}})
