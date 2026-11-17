"""Write inspectable per-step visuals and numeric results; no model dependencies."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize


def overlay_scribble(image: Image.Image, points: np.ndarray, color=(240, 49, 55), width=3):
    image = image.convert('RGB').copy()
    d = ImageDraw.Draw(image)
    d.line([tuple(map(float, p)) for p in points], fill=color, width=width, joint='curve')
    return image


def label_preview(image: Image.Image, step: int, sigma: float, guided: bool):
    image = image.convert('RGB').copy()
    d = ImageDraw.Draw(image)
    tag = f"Step {step:03d} | sigma={sigma:.3f} | feedback={'yes' if guided else 'no'}"
    d.rectangle((0, 0, min(image.width, len(tag)*8 + 12), 29), fill=(20, 25, 32))
    d.text((7, 7), tag, fill='white')
    return image


def junction_stats(skel: np.ndarray) -> dict:
    """Count local skeleton T-junctions and connected pieces for diagnostics.

    Uses 8-neighbor *circular group* counts, reducing false junctions on turns
    compared with the raw number of occupied 8-neighbors. These are discrete
    readouts, not a differentiable loss or formal topology guarantee.
    """
    from scipy.ndimage import label
    s = np.asarray(skel, dtype=bool)
    h,w = s.shape
    padded = np.pad(s.astype(np.uint8), 1)
    ring = [padded[0:h, 1:w+1], padded[0:h, 2:w+2],
            padded[1:h+1, 2:w+2], padded[2:h+2, 2:w+2],
            padded[2:h+2, 1:w+1], padded[2:h+2, 0:w],
            padded[1:h+1, 0:w], padded[0:h, 0:w]]
    groups = sum((a != 0) & (b == 0) for a,b in zip(ring, ring[-1:] + ring[:-1]))
    return dict(hard_branchpoints=int((s & (groups >= 3)).sum()),
                hard_endpoints=int((s & (groups == 1)).sum()),
                hard_connected_components=int(label(s, structure=np.ones((3,3)))[1]))


def hard_skeleton_metrics(prob_crop: np.ndarray, target_xy: np.ndarray, rect: tuple[int,int,int,int]):
    """Diagnostic-only physical-pixel distances for predicted vs target skeleton."""
    from skimage.draw import line
    h,w = prob_crop.shape
    binary = np.asarray(prob_crop > 0.5, bool)
    skel = skeletonize(binary)
    x0,y0,x1,y1 = rect
    sx,sy = (x1-x0)/w,(y1-y0)/h
    dst = np.zeros((h,w), bool)
    pts = np.rint(np.stack([(target_xy[:,0]-x0)/sx,
                             (target_xy[:,1]-y0)/sy],axis=1)).astype(int)
    pts[:,0] = np.clip(pts[:,0],0,w-1)
    pts[:,1] = np.clip(pts[:,1],0,h-1)
    for a,b in zip(pts[:-1],pts[1:]):
        rr,cc = line(a[1],a[0],b[1],b[0])
        dst[rr,cc] = True
    stats=junction_stats(skel)
    if not skel.any():
        return dict(hard_skeleton_to_stroke_px=None, hard_stroke_to_skeleton_px=None,
                    hard_skeleton_pixels=0, hard_foreground_pixels=int(binary.sum()), **stats)
    d1 = distance_transform_edt(~dst, sampling=(sy,sx))
    d2 = distance_transform_edt(~skel, sampling=(sy,sx))
    return dict(hard_skeleton_to_stroke_px=float(d1[skel].mean()),
                hard_stroke_to_skeleton_px=float(d2[dst].mean()),
                hard_skeleton_pixels=int(skel.sum()),
                hard_foreground_pixels=int(binary.sum()),**stats)


class RunRecorder:
    def __init__(self, run_dir: Path, source: Image.Image, stroke_xy: np.ndarray):
        self.root = Path(run_dir)
        self.steps = self.root / 'steps'
        self.steps.mkdir(parents=True, exist_ok=True)
        self.source = source
        self.stroke = np.asarray(stroke_xy, dtype=float)
        self.history = []
        self.gallery = []

    def record(self, step: int, sigma: float, predicted: Image.Image, probability,
               rect, metrics: dict, guided: bool, soft_skeleton=None,
               update_magnitude=None, raw_clipseg=None, change_prior=None):
        """Images show CLEAN estimate BEFORE applying this step's gradient update."""
        step_id = f'step_{step:03d}'
        predicted = predicted.convert('RGB')
        p = np.asarray(probability, dtype=np.float32)
        if p.ndim != 2:
            raise ValueError('CLIPSeg output must be H,W')
        p = np.clip(p,0,1)
        raw = np.clip(np.asarray(raw_clipseg if raw_clipseg is not None else p, dtype=np.float32), 0, 1)
        changed = np.clip(np.asarray(change_prior if change_prior is not None else np.zeros_like(p), dtype=np.float32), 0, 1)
        h,w = p.shape
        x0,y0,x1,y1 = rect
        actual = hard_skeleton_metrics(p,self.stroke,rect)
        binary_skel = skeletonize(p > .5)
        prob_image = Image.fromarray(np.uint8(np.rint(p*255)), 'L')
        raw_image = Image.fromarray(np.uint8(np.rint(raw*255)), 'L')
        changed_image = Image.fromarray(np.uint8(np.rint(changed*255)), 'L')
        skel_image = Image.new('RGB', (w,h), (19,24,33))
        a = np.asarray(skel_image).copy()
        a[binary_skel] = (0,245,162)
        skel_image = Image.fromarray(a)
        preview = overlay_scribble(predicted,self.stroke)
        # Map predicted skeleton in the CLIPSeg crop to the full-image diagnostic.
        stencil = Image.fromarray(np.uint8(binary_skel)*255, 'L').resize((x1-x0,y1-y0),Image.Resampling.NEAREST)
        overlay_mask = np.zeros((predicted.height,predicted.width),dtype=bool)
        overlay_mask[y0:y1,x0:x1] = np.asarray(stencil)>127
        pix=np.asarray(preview).copy()
        pix[overlay_mask] = (35,212,145)
        preview = label_preview(overlay_scribble(Image.fromarray(pix),self.stroke),step,sigma,guided)
        predicted.save(self.steps/f'{step_id}_pred.png')
        prob_image.save(self.steps/f'{step_id}_prob.png')
        raw_image.save(self.steps/f'{step_id}_raw_clipseg.png')
        changed_image.save(self.steps/f'{step_id}_change_prior.png')
        soft = np.zeros_like(p) if soft_skeleton is None else np.asarray(soft_skeleton, dtype=np.float32)
        soft = np.asarray(Image.fromarray(soft, 'F').resize((w,h),Image.Resampling.BILINEAR))
        if soft.max() > 0: soft=soft/soft.max()
        Image.fromarray(np.uint8(np.clip(soft*255,0,255)),'L').save(self.steps/f'{step_id}_soft_skeleton.png')
        update = np.zeros((predicted.height,predicted.width),np.float32) if update_magnitude is None else np.asarray(update_magnitude,dtype=np.float32)
        if update.shape != (predicted.height,predicted.width):
            update = np.asarray(Image.fromarray(update,'F').resize(predicted.size,Image.Resampling.BILINEAR))
        if update.max() > 0: update=update/update.max()
        Image.fromarray(np.uint8(np.clip(update*255,0,255)),'L').save(self.steps/f'{step_id}_update.png')
        skel_image.save(self.steps/f'{step_id}_skeleton.png')
        preview.save(self.steps/f'{step_id}_overlay.png')
        raw_stats=hard_skeleton_metrics(raw,self.stroke,rect)
        record = dict(step=int(step),sigma=float(sigma),guided=bool(guided),
                      **metrics,**actual,
                      raw_clipseg_hard_skeleton_pixels=raw_stats['hard_skeleton_pixels'],
                      raw_clipseg_hard_skeleton_to_stroke_px=raw_stats['hard_skeleton_to_stroke_px'],
                      raw_clipseg_hard_stroke_to_skeleton_px=raw_stats['hard_stroke_to_skeleton_px'])
        if actual['hard_skeleton_to_stroke_px'] is not None:
            record['hard_mean_bidirectional_px'] = .5 * (
                actual['hard_skeleton_to_stroke_px'] + actual['hard_stroke_to_skeleton_px'])
        else:
            record['hard_mean_bidirectional_px'] = None
        (self.steps/f'{step_id}.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        self.history.append(record)
        self.gallery.append((str(self.steps/f'{step_id}_overlay.png'),f'Step {step} · sigma {sigma:.3f}'))

    def finish(self, final: Image.Image, meta: dict):
        final.convert('RGB').save(self.root/'final.png')
        (self.root/'config.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
        (self.root/'metrics.json').write_text(json.dumps(self.history,indent=2),encoding='utf-8')
        if self.history:
            columns = list(dict.fromkeys(k for row in self.history for k in row))
            with open(self.root/'metrics.csv','w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=columns)
                writer.writeheader();writer.writerows(self.history)
        self.make_plot()
        self.make_alignment_plot()
        archive=self.root/'diagnostics.zip'
        with ZipFile(archive,'w',ZIP_DEFLATED) as z:
            for path in self.root.rglob('*'):
                if path.is_file() and path != archive:
                    z.write(path,path.relative_to(self.root))
        return archive

    def make_plot(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        steps=[r['step'] for r in self.history]
        fig,ax=plt.subplots(figsize=(8.6,4.5))
        for metric,label in [('loss_total','Total loss'),('loss_skeleton_to_stroke','Skeleton → stroke'),
                             ('loss_stroke_to_skeleton','Stroke → skeleton')]:
            series=[r.get(metric) for r in self.history]
            ax.plot(steps,series,marker='.',label=label)
        for r in self.history:
            if r.get('feedback_applied'):
                ax.axvline(r['step'],alpha=.12)
        ax.set(xlabel='Denoising step (pre-update)',ylabel='Differentiable surrogate loss',title='Step-by-step geometry feedback')
        ax.legend(loc='best');fig.tight_layout()
        fig.savefig(self.root/'loss.png',dpi=150)
        plt.close(fig)

    def make_alignment_plot(self):
        """Physical pixel alignment is independent of surrogate loss values."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        steps=[r['step'] for r in self.history]
        fig, ax=plt.subplots(figsize=(9,4.5))
        for metric,label in [('hard_skeleton_to_stroke_px','Generated → drawn'),
                             ('hard_stroke_to_skeleton_px','Drawn → generated')]:
            ax.plot(steps,[r.get(metric) for r in self.history],marker='.',label=label)
        updates=[r for r in self.history if r.get('feedback_applied')]
        for i,r in enumerate(updates):
            y=r.get('post_hard_stroke_to_skeleton_px')
            if y is not None:
                ax.plot(r['step'],y,'x',color='red',markersize=7,
                        label='Immediate post-update (fixed velocity)' if i == 0 else None)
        for r in updates:
            ax.axvline(r['step'],alpha=.10)
        ax.set(xlabel='Denoising step', ylabel='Distance (original-image pixels)',
               title='Actual thresholded centerline distances (lower is better)')
        ax.legend(loc='best');fig.tight_layout()
        fig.savefig(self.root/'alignment.png',dpi=150)
        plt.close(fig)


def files_for_step(run_dir: str, step: int):
    folder=Path(run_dir)/'steps'
    stem=f'step_{int(step):03d}'
    paths=[folder/f'{stem}_{name}.png' for name in ('pred','prob','soft_skeleton','skeleton','update','overlay','raw_clipseg','change_prior')]
    record=folder/f'{stem}.json'
    if not all(p.is_file() for p in paths) or not record.is_file():
        raise FileNotFoundError(f'Step {step} not saved under {folder}')
    return tuple(Image.open(p).copy() for p in paths), json.loads(record.read_text())
