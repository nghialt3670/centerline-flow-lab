"""One-pass FLUX.2 Klein inpainting with centerline feedback and step captures.

The Klein transformer is frozen. The scheduler.step hook sees the predicted
velocity for each step, decodes x0_hat, predicts an object mask with CLIPSeg,
builds a differentiable soft skeleton and backpropagates *only* through VAE +
CLIPSeg to the current noisy latents. The next latents are corrected; Klein's
native inpaint pipeline then restores outside-mask latents as usual.

This integration is experimental and depends on the current Diffusers Klein
inpaint pipeline internal latent layout and scheduler behavior. No guarantee
of semantic correctness is made.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import gc
import numpy as np
from PIL import Image

from .geometry import CenterlineLoss, LossSettings, changed_region_prior
from .model import ClipSeg
from .diagnostics import RunRecorder
from .input import insertion_mask, resize_input, mask_from_editor, ordered_stroke


@dataclass
class RunConfig:
    object_label: str = 'a snake'
    instruction: str = 'Add one realistic snake with natural anatomy and perspective.'
    model_id: str = 'black-forest-labs/FLUX.2-klein-base-4B'
    clipseg_id: str = 'CIDAS/clipseg-rd64-refined'
    device: str = 'auto'
    steps: int = 30
    cfg: float = 4.
    inpaint_strength: float = 1.
    seed: int = 42
    max_edge: int = 768
    region_padding: int = 80
    feedback_strength: float = .75
    feedback_every: int = 2
    sigma_min: float = .10
    sigma_max: float = .98
    max_update: float = .10
    crop_context: float = .35
    loss_resolution: int = 112
    skeleton_iterations: int = 18
    branch_weight: float = .25
    coverage_weight: float = .04
    early_change_prior: bool = True
    check_post_feedback: bool = True

    def validate(self):
        if not self.object_label.strip() or not self.model_id.strip():
            raise ValueError('Object label and Klein model are required')
        if not 1 <= self.steps <= 120 or not 0 < self.inpaint_strength <= 1:
            raise ValueError('Invalid step count or inpainting strength')
        if self.feedback_strength < 0 or not 1 <= self.feedback_every <= 20:
            raise ValueError('Invalid feedback strength/frequency')
        if not 0 <= self.sigma_min < self.sigma_max <= 1.01:
            raise ValueError('Invalid sigma guidance window')
        if not 0 < self.max_update <= 1:
            raise ValueError('max_update must be in (0,1]')
        if not .10 <= self.crop_context <= 2.0:
            raise ValueError('crop_context must be in [0.1,2.0]')
        if self.region_padding < 0:
            raise ValueError('Padding cannot be negative')
        if not 0 <= self.branch_weight <= 2 or not 0 <= self.coverage_weight <= 2:
            raise ValueError('Invalid branch or coverage weight')


def select_device(choice: str) -> str:
    import torch
    if choice not in ('auto', 'mps', 'cpu', 'cuda'):
        raise ValueError('Unsupported device')
    if choice == 'auto':
        return 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if choice == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available')
    if choice == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS not available')
    return choice


def bbox_crop(points: np.ndarray, hw: tuple[int,int], context: float):
    """Square-ish CLIPSeg crop around the scribble plus object context."""
    h, w = hw
    xy=np.asarray(points)
    if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<2:
        raise ValueError('Need an ordered centerline [N,2]')
    cx,cy = xy.mean(axis=0)
    side=max(float(np.ptp(xy[:,0])), float(np.ptp(xy[:,1])), 48.)*(1+2*context)
    side=int(np.ceil(side))
    side=min(max(h,w),side)
    x0=max(0,min(w-min(side,w),round(cx-side/2)))
    y0=max(0,min(h-min(side,h),round(cy-side/2)))
    return int(x0),int(y0),int(min(w,x0+side)),int(min(h,y0+side))


def normalized_crop_points(points: np.ndarray, rect):
    x0,y0,x1,y1=rect
    xy=np.asarray(points,dtype=np.float32).copy()
    xy[:,0]=(xy[:,0]-x0)/max(x1-x0-1,1)
    xy[:,1]=(xy[:,1]-y0)/max(y1-y0-1,1)
    return np.clip(xy,0,1)


def normalized_gradient(grad, token_gate):
    import torch
    g=grad.float()*token_gate.float()
    rms=torch.sqrt((g.square().mean()).clamp_min(1e-20))
    if float(rms.detach().cpu()) < 1e-10:
        return torch.zeros_like(grad), 0.
    out=(g/rms).clamp(-3,3)*token_gate.float()
    return out.to(grad.dtype),float(rms.detach().cpu())


class SamplerInspector:
    def __init__(self, pipe, segmenter, recorder: RunRecorder, stroke: np.ndarray,
                 edit_mask: Image.Image, device: str, config: RunConfig):
        import torch
        import torch.nn.functional as F
        self.pipe=pipe
        self.segmenter=segmenter
        self.recorder=recorder
        self.config=config
        self.stroke=stroke
        self.device=device
        self.h,self.w=recorder.source.height,recorder.source.width
        self.rect=bbox_crop(stroke,(self.h,self.w),config.crop_context)
        self.loss_fn=CenterlineLoss(normalized_crop_points(stroke,self.rect),device,
                                    LossSettings(resolution=config.loss_resolution,
                                                 skeleton_iterations=config.skeleton_iterations,
                                                 branch_weight=config.branch_weight,
                                                 coverage_weight=config.coverage_weight))
        self.mask=np.asarray(edit_mask.convert('L'),dtype=np.float32)/255
        self.source_tensor=torch.from_numpy(np.asarray(recorder.source, dtype=np.float32).copy()).permute(2,0,1)[None].to(device)/255
        self.edit_tensor=torch.from_numpy(self.mask.copy())[None,None].to(device)
        self.orig_step=None
        self.step_index=0
        self.guidance_steps=0
        self.on_step=None

    def __enter__(self):
        self.orig_step=self.pipe.scheduler.step
        self.pipe.scheduler.step=self.step
        return self

    def __exit__(self, typ, val, tb):
        self.pipe.scheduler.step=self.orig_step
        return False

    def decode_predicted_x0(self, tokens):
        """Match Diffusers Klein's packed BN-normalized VAE decode path.

        `tokens` is [B,N,C] with N=(H/16)*(W/16) for current Klein VAE.
        """
        import torch
        import torch.nn.functional as F
        p=self.pipe
        ht,wt=self.h//(p.vae_scale_factor*2),self.w//(p.vae_scale_factor*2)
        b,n,c=tokens.shape
        if ht*wt!=n or c%4:
            raise RuntimeError(f'Unsupported Klein latent layout {tokens.shape} (expected {ht*wt} packed tokens)')
        packed=tokens.transpose(1,2).reshape(b,c,ht,wt).to(dtype=p.vae.dtype)
        mean=p.vae.bn.running_mean.view(1,-1,1,1).to(packed)
        std=torch.sqrt(p.vae.bn.running_var.view(1,-1,1,1).to(packed)+p.vae.config.batch_norm_eps)
        latent=p._unpatchify_latents(packed*std+mean)
        decoded=p.vae.decode(latent,return_dict=False)[0]
        img=(decoded.float()/2+.5).clamp(0,1)
        if img.shape[-2:]!=(self.h,self.w):
            img=F.interpolate(img,(self.h,self.w),mode='bilinear',align_corners=False)
        return img

    def _token_gate(self, sample):
        import torch
        import torch.nn.functional as F
        ht,wt=self.h//(self.pipe.vae_scale_factor*2),self.w//(self.pipe.vae_scale_factor*2)
        if ht*wt!=sample.shape[1]:
            raise ValueError('Mask token grid does not match sampled latents')
        m=torch.from_numpy(self.mask)[None,None].to(sample.device)
        # Avoid mode="area" here too: it dispatches to unsupported adaptive
        # pooling on MPS when the spatial dimensions are not divisible.
        m=F.interpolate(m,(ht,wt),mode='bilinear',align_corners=False)
        return m.flatten(2).transpose(1,2).to(sample.dtype)

    def step(self, model_output, timestep, sample, *args, **kwargs):
        import torch
        # Run the unchanged scheduler first; its step_index then points to the
        # next entry, enabling correct sigma of the input and output latents.
        out=self.orig_step(model_output,timestep,sample,*args,**kwargs)
        idx=int(self.pipe.scheduler.step_index)-1
        sigma=float(self.pipe.scheduler.sigmas[idx].detach().cpu())
        sigma_next=float(self.pipe.scheduler.sigmas[idx+1].detach().cpu())
        index=self.step_index+1
        should = (self.config.feedback_strength>0 and
                  self.step_index%self.config.feedback_every==0 and
                  self.config.sigma_min<=sigma<=self.config.sigma_max)
        # The Diffusers pipeline is run under no_grad; some inference wrappers
        # additionally enable inference_mode, which enable_grad alone cannot
        # override. Build a *fresh normal Tensor leaf* before VAE/CLIPSeg.
        # The transformer velocity is deliberately stop-gradient.
        with torch.inference_mode(False):
            with torch.set_grad_enabled(should):
                x = sample.detach().to(dtype=torch.float32).clone().requires_grad_(should)
                v = model_output.detach().to(dtype=torch.float32).clone()
                x0 = x - sigma * v
                image = self.decode_predicted_x0(x0)
                logits = self.segmenter.logits(image, self.rect)
                if self.config.early_change_prior:
                    prior, change_mean = changed_region_prior(
                        image, self.source_tensor, self.rect, self.edit_tensor,
                        resolution=self.config.loss_resolution,
                    )
                else:
                    prior, change_mean = None, 0.
                loss, metrics, p, skel = self.loss_fn(logits, change_prior=prior)
                metrics['mean_source_difference'] = change_mean
                metrics['used_change_prior'] = bool(prior is not None)
                if should:
                    graph = {
                        'latent': x.requires_grad,
                        'clean_estimate': x0.requires_grad,
                        'decoded_image': image.requires_grad,
                        'clipseg_logits': logits.requires_grad,
                        'centerline_loss': loss.requires_grad,
                    }
                    if not loss.requires_grad or not logits.requires_grad:
                        raise RuntimeError(
                            f'Gradient path disconnected at step {index}: {graph}. '
                            'The VAE, CLIPSeg, and soft-skeleton loss must remain '
                            'differentiable with respect to the sampled latent.'
                        )
                    gradient = torch.autograd.grad(loss, x, retain_graph=False,
                                                   create_graph=False, allow_unused=False)[0]
                    if gradient is None or not torch.isfinite(gradient).all():
                        raise RuntimeError(f'Invalid latent gradient at step {index}: {graph}')
        update_vis = None
        metrics.update(post_loss_total=None, post_hard_skeleton_to_stroke_px=None,
                       post_hard_stroke_to_skeleton_px=None, post_loss_delta=None)
        if should:
            gate=self._token_gate(sample)
            normed,grad_rms=normalized_gradient(gradient.detach(),gate)
            eta=min(float(self.config.max_update),
                    float(self.config.feedback_strength)*max(0.,sigma-sigma_next))
            delta=-eta*normed
            # Spatial distribution of the latent correction, for diagnostics.
            ht,wt=self.h//(self.pipe.vae_scale_factor*2),self.w//(self.pipe.vae_scale_factor*2)
            magn=delta.detach().float().square().mean(-1).sqrt().reshape(ht,wt).cpu().numpy()
            update_vis=magn
            # The correction is a sampled value, not a new computation graph
            # to retain across later Klein denoising steps.
            with torch.no_grad():
                if isinstance(out,tuple):
                    out=(out[0]+delta.detach().to(out[0]),*out[1:])
                else:
                    out.prev_sample=out.prev_sample+delta.detach().to(out.prev_sample)
            self.guidance_steps+=bool(eta>0 and grad_rms>0)
            if self.config.check_post_feedback and eta > 0 and grad_rms > 0:
                # Reuse the previous frozen model velocity, not a second
                # transformer call. For an Euler scheduler, this measures the
                # immediate effect of delta on x0, NOT the next real step.
                from .diagnostics import hard_skeleton_metrics
                with torch.inference_mode(False), torch.no_grad():
                    next_latents = out[0] if isinstance(out, tuple) else out.prev_sample
                    post_clean = next_latents.detach().float().clone() - sigma_next * model_output.detach().float().clone()
                    post_img = self.decode_predicted_x0(post_clean)
                    post_logits = self.segmenter.logits(post_img, self.rect)
                    if self.config.early_change_prior:
                        post_prior, _ = changed_region_prior(
                            post_img, self.source_tensor, self.rect, self.edit_tensor,
                            resolution=self.config.loss_resolution,
                        )
                    else:
                        post_prior = None
                    post_loss, _, post_p, _ = self.loss_fn(post_logits, change_prior=post_prior)
                    from torch.nn import functional as F
                    post_np = F.interpolate(post_p, size=post_logits.shape[-2:],
                                            mode='bilinear', align_corners=False)[0,0].cpu().numpy()
                    hard_post = hard_skeleton_metrics(post_np, self.stroke, self.rect)
                    metrics.update(post_loss_total=float(post_loss.cpu()),
                                   post_loss_delta=float(post_loss.cpu()) - metrics['loss_total'],
                                   post_hard_skeleton_to_stroke_px=hard_post['hard_skeleton_to_stroke_px'],
                                   post_hard_stroke_to_skeleton_px=hard_post['hard_stroke_to_skeleton_px'])
            metrics.update(gradient_rms=grad_rms,update_rms=float(delta.float().square().mean().sqrt().cpu()),
                           step_size=eta)
        else:
            metrics.update(gradient_rms=0.,update_rms=0.,step_size=0.)
        metrics.update(sigma_next=sigma_next, skeleton_loss_kind='symmetric_soft_chamfer_plus_branch',
                       feedback_applied=bool(should and metrics['update_rms']>0))
        with torch.no_grad():
            img_np=(image[0].detach().float().cpu().permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8)
            predicted=Image.fromarray(img_np,'RGB')
            proba=p[0,0].detach().cpu().numpy()
            # Diagnostic uses probability in the FULL CLIPSeg crop grid for
            # preservation of fine detail; logits may be 352x352.
            full_prob=torch.sigmoid(logits[0,0].detach().float()).cpu().numpy()
        soft_skeleton=skel[0,0].detach().float().cpu().numpy()
        effective_prob=p[0,0].detach().float().cpu().numpy()
        change_np=(prior[0,0].detach().float().cpu().numpy()
                   if prior is not None else np.zeros_like(effective_prob))
        self.recorder.record(index,sigma,predicted,effective_prob,self.rect,metrics,
                             metrics['feedback_applied'],soft_skeleton=soft_skeleton,
                             update_magnitude=update_vis, raw_clipseg=full_prob,
                             change_prior=change_np)
        if self.on_step is not None:
            self.on_step(index,self.config.steps)
        self.step_index+=1
        return out


def run_generation(editor: dict, conf: RunConfig, root: Path, on_step=None):
    """Save all artifacts in root, return final image, recorder, config metadata."""
    import torch
    from diffusers import Flux2KleinInpaintPipeline
    conf.validate()
    source,stroke_mask=mask_from_editor(editor)
    source,stroke_mask=resize_input(source,stroke_mask,conf.max_edge)
    ordered,warning=ordered_stroke(stroke_mask,count=64)
    edit=insertion_mask(stroke_mask,int(conf.region_padding))
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    source.save(root/'source.png')
    Image.fromarray(stroke_mask.astype(np.uint8)*255,'L').save(root/'scribble.png')
    edit.save(root/'inpaint_mask.png')
    np.save(root/'scribble_points.npy',ordered)
    from .diagnostics import overlay_scribble
    overlay_scribble(source,ordered).save(root/'scribble_overlay.png')
    device=select_device(conf.device)
    dtype=torch.bfloat16 if device in ('cuda','mps') else torch.float32
    pipe=None
    recorder=RunRecorder(root,source,ordered)
    try:
        pipe=Flux2KleinInpaintPipeline.from_pretrained(conf.model_id,torch_dtype=dtype)
        if device=='cuda':
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        pipe.vae.eval().requires_grad_(False)
        segmenter=ClipSeg(conf.clipseg_id,conf.object_label,device)
        if conf.feedback_strength > 0:
            # Run CLIPSeg's backward pass once before the expensive Klein loop.
            segmenter.preflight_gradient()
        inspector=SamplerInspector(pipe,segmenter,recorder,ordered,edit,device,conf)
        inspector.on_step=on_step
        generator=torch.Generator(device='cpu').manual_seed(int(conf.seed))
        import re
        noun = re.sub(r'^(a|an|the)\s+', '', conf.object_label.strip(), flags=re.IGNORECASE)
        prompt=(f'Edit this photo: insert exactly one photorealistic {noun} '
                f'in the edit area. {conf.instruction} Natural anatomy, material, lighting and perspective. '
                'Preserve the existing scene outside the edit area.')
        with inspector:
            result=pipe(image=source,mask_image=edit,prompt=prompt,width=source.width,height=source.height,
                        num_inference_steps=int(conf.steps),guidance_scale=(float(conf.cfg) if 'base' in conf.model_id.lower() else 1.),
                        strength=float(conf.inpaint_strength),generator=generator).images[0].convert('RGB')
        from dataclasses import asdict
        meta=asdict(conf)
        meta.update(actual_device=device, actual_steps=inspector.step_index, gradient_steps=inspector.guidance_steps,
                    scribble_warning=warning, source_size=list(source.size), target_kind='open_centerline',
                    intermediate_kind='predicted_clean_x0_before_gradient',
                    model_parameters_updated=False,
                    final_is_raw=True,
                    implementation_warning='Early image-change proxy is NOT semantic segmentation, and may react to shadows. '
                        'Intermediate predictions can be unrecognizable. Mac MPS inference has not been verified.')
        archive=recorder.finish(result,meta)
        return result,recorder,meta,archive
    finally:
        if pipe is not None:
            del pipe
        gc.collect()
        if device=='cuda' and torch.cuda.is_available():torch.cuda.empty_cache()
        if device=='mps' and torch.backends.mps.is_available():torch.mps.empty_cache()
