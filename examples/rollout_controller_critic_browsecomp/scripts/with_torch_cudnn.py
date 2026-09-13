"""Run a command with the cuDNN libraries bundled alongside PyTorch.

Use inside the SIF, before importing Torch or Transformer Engine:
    python with_torch_cudnn.py python train_slime.py ...
Apply the same wrapper to Ray node startup so its workers inherit the path.
This is opt-in; it does not alter already running or queued evaluations.
"""
import importlib.util
import os
from pathlib import Path
import sys


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Usage: with_torch_cudnn.py COMMAND [ARG ...]')
    spec = importlib.util.find_spec('nvidia.cudnn')
    candidates = [] if spec is None else [
        Path(p)/'lib' for p in spec.submodule_search_locations or ()
        if (Path(p)/'lib/libcudnn.so.9').is_file()]
    if len(candidates) != 1:
        raise RuntimeError(f'Expected one bundled cuDNN library directory, found {candidates}')
    library = str(candidates[0])
    paths = [library, *os.environ.get('LD_LIBRARY_PATH', '').split(os.pathsep)]
    # TE explicitly searches CUDNN_HOME / CUDA_HOME before LD_LIBRARY_PATH.
    # Set its documented root as well; the linker path alone is insufficient.
    env = dict(os.environ, CUDNN_HOME=str(candidates[0].parent),
               LD_LIBRARY_PATH=os.pathsep.join(dict.fromkeys(p for p in paths if p)))
    os.execvpe(sys.argv[1], sys.argv[1:], env)


if __name__ == '__main__':
    main()
