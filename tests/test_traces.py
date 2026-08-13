#Round-trip: save_trace -> load_trace -> replay must reproduce the run exactly.
from pathlib import Path

import pytest
import torch

from hope_implementation.models import load_frozen
from hope_implementation.state import build_state, compress_to_density
from hope_implementation.trace import save_trace, load_trace, replay

CKPT = Path(__file__).parents[1] / "checkpoints" / "vgg8_baseline.pt"
pytestmark = pytest.mark.skipif(not CKPT.exists(), reason="baseline checkpoint not present")


def test_trace_round_trip(tmp_path):
    model = load_frozen(str(CKPT), device=torch.device("cpu"))
    cs = build_state(model)
    compress_to_density(cs, 0.97)
    assert cs.history, "no actions recorded; nothing to round-trip"

    path = tmp_path / "run.json"
    save_trace(cs, path, meta={"ckpt": str(CKPT), "target": 0.97})

    meta, acts = load_trace(path)
    assert meta["target"] == 0.97
    assert len(acts) == len(cs.history)
    for got, want in zip(acts, cs.history):
        assert (got.kind, got.layer, got.i, got.j) == (want.kind, want.layer, want.i, want.j)

    # replay from a fresh state built off the same checkpoint
    fresh = build_state(load_frozen(str(CKPT), device=torch.device("cpu")))
    replay(fresh, acts)
    assert fresh.density() == pytest.approx(cs.density(), abs=1e-12)
    for a, b in zip(fresh.states, cs.states):
        assert torch.equal(a.alive, b.alive)
        assert a.e_rem == pytest.approx(b.e_rem, rel=1e-10)