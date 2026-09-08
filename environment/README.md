# Slime without Docker: SIF + uv

The immutable Apptainer image provides CUDA 12.9, PyTorch, Megatron, Transformer
Engine, and compiled attention kernels. A uv-managed `.venv` adds this editable
checkout and inherits the image's Python packages. This avoids rebuilding or
mixing the native stack with the host TRL/vLLM environments.

```bash
bash scripts/setup_sif_env.sh
bash scripts/sif.sh python environment/check_env.py --gpu
bash scripts/sif.sh python train.py --help
bash scripts/sif.sh /opt/slime-uv pip list
```

Run `.venv/bin/python` through `scripts/sif.sh`: its base interpreter and native
libraries live in the image. Host activation alone is not sufficient. Setup
verifies the image SHA-256 and writes the actual versions to
`.venv/environment.json`. The image digest pins the complete native dependency
stack; `setup_sif_env.sh` installs only the local editable package, with no
dependency upgrades. Re-running setup is safe.

`uv pip list` shows the overlay; `environment/check_env.py` checks the actual
combined import environment, including Slime's declared runtime requirements.
`uv pip check` alone does not include the inherited SIF distributions.

The default shared image is
`/weka/projects/bvandur1/zjiang31/slime/images/slime-latest.sif`.
Set `SLIME_SIF=/path/to/copied.sif` to relocate that same image. Its contents must
match `sif-image.sha256`. Prerequisites on the host are Apptainer, uv, the NVIDIA
driver, and read access to the image and checkout. The launcher binds the
checkout and `/weka`, preserves explicit GPU/Slurm/NCCL settings, and uses a clean
container environment. Additional cluster mounts can be added to `sif.sh`.

For the existing allocation, run the same command through `srun --jobid=321337
--overlap --nodes=1 --ntasks=1 --cpu-bind=none --nodelist=gh202` after its current
GPU work finishes. The completed GRPO experiment used all four gh202 GPUs for
training and six Slime-managed SGLang engines on gh130, gh121, and gh203 for
rollouts. Slime owned synchronization of every policy update to those engines.
