# Centerline Flow Lab v0.1.4

Experimental, **training-free** centerline-guided object insertion using FLUX.2 Klein inpainting and differentiable geometry feedback. One generation pass; no SAM 3, no DTW/ARAP, no warped object silhouette. **The original image sent to Klein does not include the scribble**: the scribble determines the edit mask and drives external latent corrections.

## Install / update

With `uv` and Python 3.12:

```bash
cd ~/Projects/Labs
unzip -o ~/Downloads/centerline-flow-lab-v014-mps-resize-fix.zip
cd centerline-flow-lab
uv sync --python 3.12 --extra models --extra dev
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --extra models centerline-lab
```

Open http://127.0.0.1:7860. Existing Hugging Face model weights remain in the cache; they are **not** included in this ZIP. Run tests with `uv run --extra models --extra dev pytest -q`.

**Suggested experiment:** Klein 4B Base, 512px, 30 steps, CFG 4, feedback 0.75, every 2 steps, sigma window 0.10–0.98, branch penalty 0.25, coverage term 0.04. Test with the same seed and feedback `0` for an unguided control. Start with a single *elongated* object (snake, hose, rope); a single path is not a human pose.

## MPS compatibility fix in v0.1.4

Replaces `F.interpolate(..., mode="area")` with bilinear interpolation for
non-differentiable edit-mask resampling in both the early-change prior and the
latent token gate. `area` internally uses adaptive average pooling; PyTorch
MPS rejects nondivisible crop/output dimensions (e.g. 137×155 → 112×112).
Using bilinear avoids CPU transfers and preserves the intended smooth spatial
mask gate. No change to the geometry objective or models. A regression test
exercises nondivisible shapes and asserts no area interpolation is used.

## What's new in v0.1.3

Previous results showed Klein formed a T-shaped snake around steps 10–15, but frozen CLIPSeg was nearly black until step 25. Seven feedback corrections late in sampling were too few to change the object's established geometry; the surrogate loss fell while real hard-skeleton distances increased.

This revision adds:

1. **Earlier feedback:** default maximum sigma increased from 0.85 to 0.98. Each step still records sigma and whether feedback was applied.
2. **Image-change fallback:** while CLIPSeg is weak, a differentiable, spatially gated source-vs-generated RGB difference map is blended with its prediction. The blend approaches pure CLIPSeg when CLIPSeg has strong foreground evidence. **This fallback is not object segmentation:** changes to shadows, texture, background or other objects may be mistaken for foreground, and it may hurt the image. An experimental toggle disables it.
3. **Branch penalty:** a differentiable eight-neighbor circular-group junction surrogate discourages T intersections in the predicted soft skeleton. It does not guarantee a single connected path. The foreground coverage regularizer is reduced from 0.20 to 0.04. Branch and coverage weights are exposed in the UI.
4. **Independent metrics:** per-step hard-skeleton bidirectional distances, hard junction counts/endpoints/components, CLIPSeg peak/mean, confidence proxy, and blend weight. The plots explicitly separate the surrogate loss from physical-pixel distances.
5. **Immediate post-correction check (optional):** after guidance, decode the updated next latent with the **previous frozen velocity** and re-evaluate the same surrogate and hard distances. This is a local check, **not** the actual next Klein prediction. This costs an extra VAE + CLIPSeg forward pass for each corrected step. These measurements are diagnostic only; updates are not automatically accepted/rejected based on them.
6. **More inspectable images:** predicted clean image, blended guidance probability, raw CLIPSeg probability, source-change proxy, soft skeleton, hard skeleton, latent-update magnitude, and target/geometry overlay. All are saved each step in `outputs/<run>/steps/` and packaged into the diagnostic ZIP.

The **CLIPSeg weights stay frozen**, but its vision-encoder forward is run in differentiable mode, bypassing the native `no_grad()` restriction from the upstream Transformers implementation. The only gradient target is the Klein latent. All model parameters remain frozen.

## Algorithm

Klein receives the source image (without the scribble), a **broad rectangular inpainting region** around the stroke, and an object text prompt. At each denoising step:

1. Compute current predicted clean latent `x0_hat = x_sigma - sigma * stop_grad(velocity)`, unpack Klein tokens, and decode through its frozen VAE.
2. Compute frozen CLIPSeg soft object probabilities from a crop around the scribble. Optionally compute an early soft source-change proxy and blend using a **heuristic uncalibrated CLIPSeg confidence**.
3. Apply differentiable soft skeletonization and minimize **symmetric soft-Chamfer distance** to the user's drawn centerline, plus weak foreground coverage/off-path terms and a local branch surrogate.
4. Backpropagate through VAE and CLIPSeg to the current latent, normalize and spatially gate the gradient, and add a capped correction to the scheduler's next latent.
5. Save source signals, geometric targets, corrections, and independent metrics. Optionally evaluate the same-step correction using a fixed-velocity postcheck.

All intermediate images and masks are **pre-update** estimates. `final.png` is Klein's raw output; there is no postprocessing to paste an object onto the original.

## Interpreting results

- `loss.png`: differentiable *surrogate objective*. Lower is not necessarily better image geometry.
- `alignment.png`: bidirectional hard-skeleton distances in **original image pixels**. Empty if foreground segmentation never exceeds 0.5; these are also only estimates, not human-annotated truth.
- `*_raw_clipseg.png`: semantic segmentation signal alone.
- `*_change_prior.png`: *source-difference*, not semantic segmentation.
- `*_prob.png`: **blended map used by the guidance loss**, not raw CLIPSeg.
- `*_soft_skeleton.png`: soft guidance skeleton, **normalized for visibility**; use `soft_skeleton_mass` to judge whether a bright display is actually significant.
- `*_skeleton.png`: hard thresholded skeleton, diagnostic only; `*_overlay.png` overlays the hard skeleton in green and target stroke in red.
- `metrics.csv` / `metrics.json`: per-step confidence proxy, blend weight, branch loss, independent hard-distance measurements, post-update checks, and gradient/update RMS.

**Limits:** The early change proxy can encourage undesired changes, the junction penalty does not prove topology, and CLIPSeg may not identify the generated object. A single drawn line does not fully specify width, anatomy, depth ordering, or articulation. These changes need an actual guided-vs-unguided same-seed test before claiming visual improvement. **The new v0.1.3 was unit-tested with mocks on CPU; end-to-end Klein inference and MPS gradients for this version were not run here.**

## Source layout

```text
src/centerline_lab/
  app.py          Single-screen Gradio research UI
  sampler.py      Klein scheduler hook, feedback, immediate post-check
  geometry.py     Source-change prior, soft skeleton, branch-aware loss
  model.py        Frozen CLIPSeg with image gradients preserved
  diagnostics.py  Step images, real hard geometry metrics, plots, ZIP
  input.py        Image and scribble extraction, edit mask
```
