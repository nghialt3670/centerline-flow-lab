"""Regression for unsupported adaptive average pooling on Apple MPS."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from types import SimpleNamespace

from centerline_lab.geometry import changed_region_prior
from centerline_lab.sampler import SamplerInspector


def _reject_area(monkeypatch):
    interpolate = F.interpolate
    def checked(*args, **kwargs):
        assert kwargs.get('mode') != 'area', 'MPS-incompatible area resize'
        return interpolate(*args, **kwargs)
    monkeypatch.setattr(F, 'interpolate', checked)


def test_nondivisible_crop_mask_resize_preserves_gradients(monkeypatch):
    _reject_area(monkeypatch)
    source = torch.full((1, 3, 183, 169), .5)
    generated = (source + .13 * torch.randn_like(source)).detach().requires_grad_(True)
    allowed = torch.zeros((1, 1, 183, 169))
    allowed[:, :, 19:154, 11:163] = 1.0
    rect = (9, 15, 153, 166)  # 144 x 151 -> 112 x 112
    prior, mean = changed_region_prior(generated, source, rect, allowed, 112)
    assert prior.shape == (1, 1, 112, 112)
    assert 0 <= prior.min() <= prior.max() <= 1
    assert mean > 0
    prior.sum().backward()
    assert generated.grad is not None and generated.grad.abs().sum() > 0


def test_token_gate_does_not_require_divisible_grid(monkeypatch):
    _reject_area(monkeypatch)
    inspector = SamplerInspector.__new__(SamplerInspector)
    inspector.h, inspector.w = 520, 504
    inspector.pipe = SimpleNamespace(vae_scale_factor=8)
    inspector.mask = np.ones((520, 504), dtype=np.float32)
    sample = torch.zeros((1, (520//16)*(504//16), 4))
    gate = inspector._token_gate(sample)
    assert gate.shape == (1, sample.shape[1], 1)
    assert torch.all(gate == 1)


@pytest.mark.skipif(not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available(),
                    reason='Requires Apple MPS')
def test_changed_region_prior_nondivisible_on_actual_mps():
    device = 'mps'
    source = torch.full((1, 3, 181, 167), .5, device=device)
    generated = (source + .11 * torch.rand_like(source)).detach().requires_grad_(True)
    allowed = torch.ones((1, 1, 181, 167), device=device)
    prior, mean = changed_region_prior(generated, source, (7, 11, 157, 169), allowed, 112)
    prior.mean().backward()
    assert generated.grad is not None and torch.isfinite(generated.grad).all()
    assert mean > 0
