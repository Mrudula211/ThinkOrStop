import numpy as np
import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from think_or_stop.inference import adaptive_predict, simulate_adaptive


def _reference(model, x, thr):
    """Straightforward per-sample implementation of the exit rule."""
    preds, exits = [], []
    with torch.no_grad():
        for xi in x:
            outs = model(xi.unsqueeze(0))
            for e, z in enumerate(outs):
                p = F.softmax(z, 1)
                c, k = p.max(1)
                if e == len(outs) - 1 or c.item() >= thr:
                    preds.append(k.item())
                    exits.append(e)
                    break
    return np.array(preds), np.array(exits)


def _probs(model, x):
    with torch.no_grad():
        return torch.stack([F.softmax(z, 1) for z in model(x)], 1).numpy()


def _mid_threshold(model, x):
    """A threshold that splits the batch so all three exits are used."""
    p = _probs(model, x)
    return float(np.median(p[:, 0].max(1)))


def test_single_sample(small_multiexit):
    x = torch.randn(3, 32, 32)
    r = adaptive_predict(small_multiexit, x, 0.5)
    assert r.predictions.shape == (1,) and r.exit_indices.shape == (1,)
    assert 0 <= int(r.predictions) < 10 and 0 <= int(r.exit_indices) <= 2


def test_batched_matches_per_sample_reference_and_order(small_multiexit):
    torch.manual_seed(1)
    x = torch.randn(64, 3, 32, 32)
    thr = _mid_threshold(small_multiexit, x)
    r = adaptive_predict(small_multiexit, x, thr)
    ref_pred, ref_exit = _reference(small_multiexit, x, thr)
    # Each position must hold its own sample's result after dynamic regrouping.
    np.testing.assert_array_equal(r.exit_indices.numpy(), ref_exit)
    np.testing.assert_array_equal(r.predictions.numpy(), ref_pred)
    assert len(set(ref_exit.tolist())) >= 2  # the batch really was split
    # The cached-logit simulation agrees with the dynamic implementation.
    sim = simulate_adaptive(_probs(small_multiexit, x), thr)
    np.testing.assert_array_equal(sim.exit_indices, ref_exit)
    np.testing.assert_array_equal(sim.predictions, ref_pred)


def test_order_with_shuffled_batch(small_multiexit):
    torch.manual_seed(2)
    x = torch.randn(40, 3, 32, 32)
    thr = _mid_threshold(small_multiexit, x)
    perm = torch.randperm(40)
    a = adaptive_predict(small_multiexit, x, thr)
    b = adaptive_predict(small_multiexit, x[perm], thr)
    torch.testing.assert_close(b.predictions, a.predictions[perm])
    torch.testing.assert_close(b.exit_indices, a.exit_indices[perm])


def test_exactly_one_valid_prediction_per_sample(small_multiexit):
    x = torch.randn(33, 3, 32, 32)
    r = adaptive_predict(small_multiexit, x, _mid_threshold(small_multiexit, x))
    assert r.predictions.shape == (33,) and r.exit_indices.shape == (33,) and r.confidences.shape == (33,)
    assert r.predictions.dtype == torch.long
    assert ((r.predictions >= 0) & (r.predictions < 10)).all()
    assert ((r.exit_indices >= 0) & (r.exit_indices <= 2)).all()
    assert ((r.confidences > 0) & (r.confidences <= 1)).all()


def test_exited_samples_skip_deeper_stages(small_multiexit):
    seen = []
    hooks = [s.register_forward_hook(lambda m, i, o, k=k: seen.append((k, i[0].shape[0])))
             for k, s in enumerate(small_multiexit.stages)]
    try:
        x = torch.randn(50, 3, 32, 32)
        thr = _mid_threshold(small_multiexit, x)
        r = adaptive_predict(small_multiexit, x, thr)
        n1 = int((r.exit_indices == 0).sum())
        n2 = int((r.exit_indices == 1).sum())
        assert dict(seen) == {0: 50, 1: 50 - n1, 2: 50 - n1 - n2}
        assert r.stage_batch_sizes == [50, 50 - n1, 50 - n1 - n2]

        seen.clear()
        r = adaptive_predict(small_multiexit, x, 0.0)  # everything exits at exit 1
        assert dict(seen) == {0: 50}
        assert (r.exit_indices == 0).all()

        seen.clear()
        r = adaptive_predict(small_multiexit, x, float("inf"))  # nothing exits early
        assert dict(seen) == {0: 50, 1: 50, 2: 50}
        assert (r.exit_indices == 2).all()
    finally:
        for h in hooks:
            h.remove()


def test_deterministic_and_no_gradients(small_multiexit):
    x = torch.randn(20, 3, 32, 32, requires_grad=True)
    a = adaptive_predict(small_multiexit, x, 0.3)
    b = adaptive_predict(small_multiexit, x, 0.3)
    assert torch.equal(a.predictions, b.predictions) and torch.equal(a.exit_indices, b.exit_indices)
    assert torch.equal(a.confidences, b.confidences)
    assert not a.confidences.requires_grad and torch.is_inference(a.confidences)
    assert all(p.grad is None for p in small_multiexit.parameters())


def test_requires_eval_mode(small_multiexit):
    small_multiexit.train()
    with pytest.raises(RuntimeError, match="eval"):
        adaptive_predict(small_multiexit, torch.randn(2, 3, 32, 32), 0.9)


def test_incomplete_final_batch(small_multiexit):
    x = torch.randn(23, 3, 32, 32)
    loader = DataLoader(TensorDataset(x, torch.zeros(23, dtype=torch.long)), batch_size=8)
    thr = _mid_threshold(small_multiexit, x)
    parts = [adaptive_predict(small_multiexit, xb, thr) for xb, _ in loader]
    assert [p.predictions.numel() for p in parts] == [8, 8, 7]
    full = adaptive_predict(small_multiexit, x, thr)
    torch.testing.assert_close(torch.cat([p.predictions for p in parts]), full.predictions)


def test_per_exit_thresholds(small_multiexit):
    x = torch.randn(10, 3, 32, 32)
    r = adaptive_predict(small_multiexit, x, [float("inf"), 0.0])  # skip exit 1, always take exit 2
    assert (r.exit_indices == 1).all()
    with pytest.raises(ValueError):
        adaptive_predict(small_multiexit, x, [0.5, 0.5, 0.5])


def test_baseline_always_full_depth(small_baseline):
    r = adaptive_predict(small_baseline, torch.randn(6, 3, 32, 32), 0.0)
    assert (r.exit_indices == 0).all() and r.stage_batch_sizes == [6, 6, 6]
