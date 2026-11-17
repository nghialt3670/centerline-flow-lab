"""Frozen CLIPSeg with *differentiable input* for centerline guidance.

Why not call CLIPSegForImageSegmentation.forward()? In Transformers 5.x,
that method explicitly wraps the CLIP vision encoder in ``torch.no_grad()``.
Consequently the output mask is disconnected from the input image, even if
all callers have enabled gradients. We execute its equivalent vision and
segmentation-decoder operations directly, while keeping every weight frozen.

The conditional text embedding is constant and cached once per run.
"""
from __future__ import annotations


class ClipSeg:
    """Frozen text-conditioned segmentation; preserves d(logits)/d(image)."""

    def __init__(self, model_id: str, object_label: str, device: str):
        import torch
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

        self.device = torch.device(device)
        processor = CLIPSegProcessor.from_pretrained(model_id)
        self.model = CLIPSegForImageSegmentation.from_pretrained(model_id).to(self.device).eval()
        self.model.requires_grad_(False)  # Freeze weights, NOT the input image.
        tokens = processor.tokenizer(
            [object_label], padding='max_length', max_length=77,
            truncation=True, return_tensors='pt',
        )
        tokens = {
            k: v.to(self.device) for k, v in tokens.items()
            if k in ('input_ids', 'attention_mask')
        }
        # CLIPSeg's text feature is independent of the image, so caching it
        # does not detach any desired image gradient.
        with torch.no_grad():
            self.conditioning = self.model.get_conditional_embeddings(
                batch_size=1, **tokens,
            ).detach()

        ip = processor.image_processor
        size = ip.size
        self.size = int(size.get('height', size.get('shortest_edge', 352)))
        self.mean = torch.tensor(ip.image_mean, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(ip.image_std, device=self.device).view(1, 3, 1, 1)

    def _logits_from_pixels(self, pixels):
        """Exact CLIPSeg vision/decoder route, omitting its enclosing no_grad.

        For Transformers 5.x, vision_outputs.hidden_states contains the input
        embeddings at index 0 followed by the output of each encoder layer;
        extract_layers therefore use index + 1, matching HF's forward method.
        """
        import torch

        vision_outputs = self.model.clip.vision_model(
            pixel_values=pixels,
            output_hidden_states=True,
            return_dict=True,
            interpolate_pos_encoding=True,
        )
        activations = [
            vision_outputs.hidden_states[layer + 1]
            for layer in self.model.extract_layers
        ]
        decoder_outputs = self.model.decoder(
            activations, self.conditioning, return_dict=True,
        )
        logits = decoder_outputs.logits
        if logits.ndim == 3:
            logits = logits[:, None]
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise RuntimeError(f'Unexpected CLIPSeg logits shape: {tuple(logits.shape)}')
        if pixels.requires_grad and torch.is_grad_enabled() and not logits.requires_grad:
            raise RuntimeError('CLIPSeg vision/decoder did not preserve pixel gradients')
        return logits

    def logits(self, image01, rect):
        """Segmentation logits [B,1,H,W], differentiable wrt image01."""
        import torch.nn.functional as F

        x0, y0, x1, y1 = rect
        crop = image01[:, :, y0:y1, x0:x1]
        if min(crop.shape[-2:]) < 8:
            raise ValueError('Segmentation crop too small')
        crop = F.interpolate(
            crop, (self.size, self.size), mode='bilinear', align_corners=False,
        )
        crop = crop.to(device=self.device, dtype=self.mean.dtype)
        pixels = (crop - self.mean) / self.std
        return self._logits_from_pixels(pixels)

    def preflight_gradient(self):
        """Fail before Klein sampling if CLIPSeg cannot backprop to an image.

        Catches HF's internal no_grad detachment or unsupported backend
        backward operations before an expensive full denoising run.
        """
        import torch

        with torch.inference_mode(False), torch.enable_grad():
            image = torch.rand((1, 3, 32, 32), device=self.device).requires_grad_(True)
            logits = self.logits(image, (0, 0, 32, 32))
            gradient = torch.autograd.grad(logits.square().mean(), image)[0]
            if gradient is None or not torch.isfinite(gradient).all():
                raise RuntimeError(
                    'CLIPSeg input-gradient preflight failed: invalid gradient. '
                    'Check Torch / Transformers / MPS compatibility.'
                )
        return True
