"""Auto-generated shape-dispatching kl_div_forward_with_saved/kl_div_backward_from_saved.

Routes each call to the small- or large-regime specialist by regime_feature at runtime.
Threshold (on regime_feature) = 11585.237502960395.
"""
import importlib.util as _ilu
from collections import namedtuple as _nt

_SMALL_PATH = '/u/wzhan/openevolve/benchmark/triton_kl_div_backward_bench/evolve_small_r1/checkpoints/checkpoint_5/best_program.py'
_LARGE_PATH = '/u/wzhan/openevolve/benchmark/triton_kl_div_backward_bench/evolve_large_r1/checkpoints/checkpoint_5/best_program.py'
_TS_PATH = '/u/wzhan/openevolve/benchmark/triton_kl_div_backward_bench/task_spec.py'
_THRESHOLD = 11585.237502960395
_FWD = 'kl_div_forward_with_saved'
_BWD = 'kl_div_backward_from_saved'


def _load(path):
    s = _ilu.spec_from_file_location("disp_" + path.replace("/", "_"), path)
    m = _ilu.module_from_spec(s); s.loader.exec_module(m); return m


def _load_ts():
    s = _ilu.spec_from_file_location("disp_ts", _TS_PATH)
    m = _ilu.module_from_spec(s); s.loader.exec_module(m); return m


_small = _load(_SMALL_PATH)
_large = _load(_LARGE_PATH)
_ts = _load_ts()
_Case = _nt("Case", "rows cols")


def _feature(x):
    rows = 1
    for d in x.shape[:-1]:
        rows *= int(d)
    return _ts.regime_feature(_Case(rows, int(x.shape[-1])))


def kl_div_forward_with_saved(*args, **kwargs):
    # route by the primary input's regime feature (rows / numel)
    prog = _small if _feature(args[0]) < _THRESHOLD else _large
    return getattr(prog, _FWD)(*args, **kwargs)


def kl_div_backward_from_saved(dout, saved_tensors, *args, **kwargs):
    # dout (the cotangent) has the same regime-defining shape as the forward input, so it
    # routes to the same specialist that produced `saved_tensors` — no need to tag saved
    # (keeps saved a pure tensor tuple, as the evaluator requires).
    prog = _small if _feature(dout) < _THRESHOLD else _large
    return getattr(prog, _BWD)(dout, saved_tensors, *args, **kwargs)
