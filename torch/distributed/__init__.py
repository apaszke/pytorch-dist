import torch
import warnings

warnings.warn("""
================================================================================
                                    WARNING
================================================================================
torch.distributed is a highly experimental package. The API will change without
notice and we're can't guarantee full correctness and expected performance yet.
We'll announce it once it's ready.
""")


_INITIALIZED_PG = 1
_INITIALIZED_MW = 2
_initialized = 0
_scope = locals()


def _extend_scope(module):
    _scope.update({k: getattr(module, k) for k in dir(module) if not k.startswith('_')})


def init_process_group(backend):
    global _initialized
    if _initialized:
        raise RuntimeError("trying to initialize torch.distributed twice!")
    torch._C._dist_init_process_group(backend)
    _initialized = _INITIALIZED_PG
    import torch.distributed.collectives as collectives
    _extend_scope(collectives)
    assert torch._C._dist_init_extension(False, reduce_op, group)


def init_master_worker(backend):
    global _initialized
    if _initialized:
        raise RuntimeError("trying to initialize torch.distributed twice!")
    torch._C._dist_init_master_worker(backend)
    _initialized = _INITIALIZED_MW
    import torch.distributed.collectives as collectives
    import torch.distributed.remote_types as remote_types
    _extend_scope(collectives)
    _extend_scope(remote_types)
    assert torch._C._dist_init_extension(True, reduce_op, group)
