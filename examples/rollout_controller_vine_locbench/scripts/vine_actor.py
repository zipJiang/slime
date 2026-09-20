"""Keep unused distributed-checkpoint padding deterministic for readback audits."""
from slime.backends.megatron_utils.actor import MegatronTrainRayActor


class VineActor(MegatronTrainRayActor):
    def init(self, args, role, with_ref=False, with_opd_teacher=False):
        from checkpoint_padding import install
        install()
        from checkpoint_restore_memory import install as install_restore_memory
        install_restore_memory()
        from checkpoint_restore_memory import install_alias_restore
        install_alias_restore()
        return super().init(args, role, with_ref, with_opd_teacher)
