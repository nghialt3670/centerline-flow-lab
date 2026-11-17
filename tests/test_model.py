"""Regression tests for HF CLIPSeg's intentional vision no_grad() block.

Use tiny fake modules to avoid downloading hundreds of MB in CI, but simulate
CLIPSeg's actual 5.x hidden_states / extract_layers / decoder API.
"""
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from centerline_lab.model import ClipSeg


class Vision(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Conv2d(3, 4, 3, padding=1)
        self.layers = nn.ModuleList([nn.Conv2d(4, 4, 1) for _ in range(3)])

    def forward(self, pixel_values, **kwargs):
        y = self.embedding(pixel_values)
        states = [y]
        for layer in self.layers:
            y = torch.tanh(layer(y))
            states.append(y)
        return SimpleNamespace(hidden_states=states)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.output = nn.Conv2d(4, 1, 1)

    def forward(self, activations, conditioning, **kwargs):
        x = sum(activations) / len(activations)
        return SimpleNamespace(logits=self.output(x).squeeze(1) + conditioning[:, 0, None, None])


class FrozenFakeClipSeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = SimpleNamespace(vision_model=Vision())
        self.decoder = Decoder()
        self.extract_layers = [0, 2]

    def forward(self, pixel_values):
        # Real HF CLIPSegForImageSegmentation.forward has this no_grad block.
        with torch.no_grad():
            vision = self.clip.vision_model(pixel_values=pixel_values, output_hidden_states=True)
            activations = [vision.hidden_states[i + 1] for i in self.extract_layers]
        return self.decoder(activations, torch.ones(1, 4)).logits


def fake_segmenter():
    seg = ClipSeg.__new__(ClipSeg)
    seg.device = torch.device('cpu')
    seg.model = FrozenFakeClipSeg().eval()
    seg.model.requires_grad_(False)
    seg.conditioning = torch.ones(1, 4)
    seg.size = 32
    seg.mean = torch.zeros(1, 3, 1, 1)
    seg.std = torch.ones(1, 3, 1, 1)
    return seg


def test_hf_default_path_breaks_gradient_but_new_path_does_not():
    seg = fake_segmenter()
    pixels = torch.rand(1, 3, 32, 32, requires_grad=True)
    default_logits = seg.model(pixel_values=pixels)
    assert not default_logits.requires_grad
    corrected_logits = seg._logits_from_pixels(pixels)
    assert corrected_logits.requires_grad
    assert corrected_logits.shape == (1, 1, 32, 32)
    assert torch.allclose(default_logits[:, None], corrected_logits, atol=1e-6)
    grad = torch.autograd.grad(corrected_logits.square().mean(), pixels)[0]
    assert grad is not None and torch.isfinite(grad).all()
    assert grad.abs().sum() > 0
    assert all(not param.requires_grad for param in seg.model.parameters())


def test_clipseg_gradient_under_no_grad_and_inference_mode():
    seg = fake_segmenter()
    with torch.no_grad(), torch.inference_mode():
        # Sampler wraps the entire feedback work in these contexts.
        with torch.inference_mode(False), torch.enable_grad():
            pixels = torch.rand(1, 3, 32, 32).requires_grad_(True)
            logits = seg.logits(pixels, (0, 0, 32, 32))
            assert logits.requires_grad
            grad = torch.autograd.grad(logits.square().mean(), pixels)[0]
            assert grad.isfinite().all()


def test_preflight_runs_backward_without_updating_weights():
    seg = fake_segmenter()
    old_params = [p.detach().clone() for p in seg.model.parameters()]
    assert seg.preflight_gradient() is True
    assert all(torch.equal(p, ref) for p, ref in zip(seg.model.parameters(), old_params))
    assert all(p.grad is None for p in seg.model.parameters())
