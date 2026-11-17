"""Differentiable geometry guidance from a soft foreground probability map.

Predicted centerlines are inferred from the current generated image, not warped
from any Stage-2 mask. A source-image residual can supply an *uncertain* early
foreground proxy when CLIPSeg is too weak to provide a useful mask. The blend
is logged so it cannot be confused with trusted segmentation.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def soft_erode(x):
    import torch
    import torch.nn.functional as F
    return torch.minimum(-F.max_pool2d(-x, (3, 1), stride=1, padding=(1, 0)),
                         -F.max_pool2d(-x, (1, 3), stride=1, padding=(0, 1)))


def soft_dilate(x):
    import torch.nn.functional as F
    return F.max_pool2d(x, 3, stride=1, padding=1)


def soft_open(x):
    return soft_dilate(soft_erode(x))


def soft_skeletonize(prob, iterations: int = 18):
    """Differentiable clDice-style morphological skeletonization."""
    import torch.nn.functional as F
    if prob.ndim != 4 or prob.shape[1] != 1:
        raise ValueError('Expected foreground probability [B,1,H,W]')
    if not 1 <= iterations <= 64:
        raise ValueError('Skeleton iterations must be 1..64')
    cur = prob
    skel = F.relu(cur - soft_open(cur))
    for _ in range(iterations):
        cur = soft_erode(cur)
        delta = F.relu(cur - soft_open(cur))
        skel = skel + F.relu(delta - skel * delta)
    return skel.clamp(0, 1)


def grid_centers(side: int, device):
    import torch
    u = torch.linspace(0, 1, side, device=device)
    yy, xx = torch.meshgrid(u, u, indexing='ij')
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), -1)


def branchpoint_surrogate(skeleton):
    """Soft junction count: penalize >2 separated neighbor groups.

    Eight neighbor directions are arranged circularly. A continuous straight
    line has two groups; a T-shaped junction has three. Smooth probabilities
    make the crossing-number test differentiable. This is a LOCAL SURROGATE,
    not a guarantee of exactly one connected centerline or valid topology.
    """
    import torch
    import torch.nn.functional as F
    if skeleton.ndim != 4 or skeleton.shape[1] != 1:
        raise ValueError('Expected skeleton [B,1,H,W]')
    # Relative display-independent activation. Detaching the scale prevents
    # encouraging all skeleton pixels to become infinitesimally faint.
    peak = skeleton.detach().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-3)
    floor = torch.sigmoid(torch.tensor(-0.22 / 0.08, device=skeleton.device))
    occupancy = ((torch.sigmoid((skeleton / peak - 0.22) / 0.08) - floor) /
                 (1 - floor)).clamp(0, 1)
    # Pad with zeros and ignore the one-pixel image boundary.
    c = F.pad(occupancy, (1, 1, 1, 1), value=0.)
    h, w = skeleton.shape[-2:]
    ring = [c[..., 0:h, 1:w+1], c[..., 0:h, 2:w+2],
            c[..., 1:h+1, 2:w+2], c[..., 2:h+2, 2:w+2],
            c[..., 2:h+2, 1:w+1], c[..., 2:h+2, 0:w],
            c[..., 1:h+1, 0:w], c[..., 0:h, 0:w]]
    groups = sum(a * (1 - b) for a, b in zip(ring, ring[-1:] + ring[:-1]))
    # Boundary pixels are excluded; padding should not create junctions.
    center = occupancy[..., 1:-1, 1:-1]
    excessive = F.relu(groups[..., 1:-1, 1:-1] - 2.05).square()
    # Emphasize sparse junctions: otherwise dividing by every skeleton pixel
    # makes a single T-branch almost invisible to the optimizer.
    return 32.0 * (excessive * center).sum(dim=(-1, -2, -3)) / center.sum(dim=(-1, -2, -3)).clamp_min(1.)


def segmentation_reliability(prob):
    """Conservative confidence proxy, NOT calibrated segmentation accuracy.

    An almost-black CLIPSeg map is unreliable for constructing a skeleton.
    The returned scalar is deliberately detached from the guidance gradient.
    """
    import torch
    import torch.nn.functional as F
    p = prob.detach().float()
    peak = p.amax(dim=(-2, -1), keepdim=True)
    # Fraction of fairly confident foreground pixels, using a smooth step.
    occupied = torch.sigmoid((p - 0.50) / 0.05).mean(dim=(-2, -1), keepdim=True)
    peak_factor = ((peak - .40) / .40).clamp(0, 1)
    area_factor = (occupied / .006).clamp(0, 1)
    return (peak_factor * area_factor).detach()


def changed_region_prior(image, source, crop_rect, edit_mask, resolution=112):
    """Differentiable source-vs-generated difference proxy for early feedback.

    This is NOT semantic segmentation: shadows or other edits may be picked up.
    All robust thresholds are detached. The editable mask gates where the
    residual is allowed to count as foreground.
    """
    import torch
    import torch.nn.functional as F
    x0, y0, x1, y1 = crop_rect
    diff = (image[:, :, y0:y1, x0:x1].float() -
            source[:, :, y0:y1, x0:x1].float()).abs().mean(1, keepdim=True)
    diff = F.interpolate(diff, (resolution, resolution), mode='bilinear', align_corners=False)
    diff = F.avg_pool2d(diff, kernel_size=5, stride=1, padding=2)
    # MPS does not support adaptive_avg_pool2d for arbitrary/nondivisible
    # input/output sizes. F.interpolate(..., mode="area") invokes that op,
    # so it fails for common crops such as 137x155 -> 112x112. The edit mask
    # is a spatial gate (not a learned signal); bilinear sampling is adequate
    # and is supported on MPS for nondivisible shapes.
    allowed = F.interpolate(
        edit_mask[:, :, y0:y1, x0:x1].float(),
        size=(resolution, resolution), mode='bilinear', align_corners=False,
    ).clamp(0, 1)
    # CPU quantiles on detached values avoid MPS quantile-autograd issues.
    valid = diff.detach()[allowed > 0.5].float().cpu().numpy()
    if valid.size < 16:
        return torch.zeros_like(diff), 0.
    q50, q75, q90 = np.quantile(valid, [.50, .75, .90])
    # If MOST of the crop changes similarly, do not label all pixels object.
    # Set the threshold strictly above the common change magnitude.
    contrast = max(.03, float(q90 - q50))
    threshold = max(.05, float(q75 + .35 * contrast))
    proxy = torch.sigmoid((diff - threshold) / max(.018, .30 * contrast)) * allowed
    return proxy, float(diff.detach()[allowed > .5].mean().cpu())


@dataclass
class LossSettings:
    resolution: int = 112
    skeleton_iterations: int = 18
    temperature: float = .035
    coverage_weight: float = .04
    offpath_weight: float = .08
    branch_weight: float = .25
    path_band: float = .10


class CenterlineLoss:
    """Soft symmetric Chamfer + differentiable local branch penalty.

    The target is a uniformly sampled, ordered centerline in normalized xy.
    The current predicted skeleton is computed from CURRENT image probabilities.
    """
    def __init__(self, normalized_xy: np.ndarray, device: str, settings: LossSettings):
        import torch
        points = np.asarray(normalized_xy, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError('Expected target points [N,2]')
        if not np.all(np.isfinite(points)) or np.min(points) < -0.001 or np.max(points) > 1.001:
            raise ValueError('Target points outside normalized crop')
        if not 32 <= settings.resolution <= 256:
            raise ValueError('Geometry resolution must be 32..256')
        if not 0 < settings.temperature < 1 or settings.path_band <= 0 or settings.branch_weight < 0:
            raise ValueError('Invalid geometry loss settings')
        self.settings = settings
        self.points = torch.from_numpy(points).to(device)
        grid = grid_centers(settings.resolution, device)
        self.distance = torch.cdist(self.points[None], grid[None])[0].clamp_min(0)
        self.distance_to_target = self.distance.min(0).values[None]
        self.far_mask = (self.distance_to_target > settings.path_band).to(self.points.dtype)

    def __call__(self, logits, change_prior=None, seg_weight=None):
        import torch
        import torch.nn.functional as F
        if logits.ndim == 3:
            logits = logits[:, None]
        if logits.ndim != 4 or logits.shape[:2] != (1, 1):
            raise ValueError('Expected one foreground logits map [1,1,H,W]')
        shape = (self.settings.resolution,) * 2
        raw_p = torch.sigmoid(F.interpolate(logits.float(), shape, mode='bilinear', align_corners=False))
        confidence = segmentation_reliability(raw_p)
        if change_prior is None:
            blend = torch.ones_like(confidence)
            p = raw_p
        else:
            if change_prior.shape != raw_p.shape:
                raise ValueError('Residual-proxy size must match geometry grid')
            blend = confidence if seg_weight is None else torch.as_tensor(seg_weight, device=raw_p.device).clamp(0, 1)
            p = (blend * raw_p + (1 - blend) * change_prior).clamp(1e-5, 1-1e-5)
        skel = soft_skeletonize(p, iterations=self.settings.skeleton_iterations)
        weights = skel.flatten(1).clamp_min(1e-8)
        pred_to_target = (weights * self.distance_to_target).sum(-1) / weights.sum(-1).clamp_min(1e-7)
        tau = self.settings.temperature
        log_w = torch.log(weights)
        numer = torch.logsumexp(log_w[:, None, :] - self.distance[None] / tau, dim=-1)
        denom = torch.logsumexp(log_w, dim=-1)[:, None]
        target_to_pred = (-tau * (numer - denom)).mean(dim=-1)
        grid = (self.points[None, None] * 2 - 1)
        sampled = F.grid_sample(p, grid, mode='bilinear', align_corners=True)[:, 0, 0]
        coverage = -torch.log(sampled.clamp_min(1e-5)).mean(dim=-1)
        outside = (p.flatten(1) * self.far_mask).sum(-1) / self.far_mask.sum(-1).clamp_min(1)
        branch = branchpoint_surrogate(skel)
        loss = (pred_to_target + target_to_pred
                + self.settings.coverage_weight * coverage
                + self.settings.offpath_weight * outside
                + self.settings.branch_weight * branch).mean()
        metrics = dict(
            loss_total=float(loss.detach().cpu()),
            loss_skeleton_to_stroke=float(pred_to_target.mean().detach().cpu()),
            loss_stroke_to_skeleton=float(target_to_pred.mean().detach().cpu()),
            loss_branch=float(branch.mean().detach().cpu()),
            loss_coverage=float(coverage.mean().detach().cpu()),
            loss_offpath=float(outside.mean().detach().cpu()),
            mean_stroke_foreground=float(sampled.mean().detach().cpu()),
            mean_foreground=float(p.mean().detach().cpu()),
            raw_clipseg_mean=float(raw_p.mean().detach().cpu()),
            raw_clipseg_peak=float(raw_p.max().detach().cpu()),
            clipseg_confidence_proxy=float(confidence.mean().detach().cpu()),
            clipseg_blend_weight=float(blend.mean().detach().cpu()),
            soft_skeleton_mass=float(skel.sum().detach().cpu()),
        )
        return loss, metrics, p.detach(), skel.detach()
