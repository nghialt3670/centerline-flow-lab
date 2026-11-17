"""Inspect FLUX.2 Klein Base image-query -> target-text-key attention.

The source image is optionally downscaled first so its long side is at most
512 pixels by default. This keeps inference and attention extraction much faster.
The downscaled image is what gets passed to Diffusers and what the attention
visualizations are drawn on.

By default, record EVERY denoising timestep for the chosen transformer layers.
At guidance_scale > 1, the Base pipeline runs conditional and unconditional
forwards: hooks record ONLY the first (positive-prompt) pass of each timestep.

Output:
  generated.png
  attention.pt                    raw per-head target-token probabilities
  attention_overview.png          mean across chosen layers, EVERY timestep
  attention_timesteps_<layer>.png  all timesteps, 10 columns, for EACH layer
  attention_heads.png             per-head view at a middle step

Usage:
  uv run python flux2_klein_attention.py \\
    --image scene.png --prompt 'Add a red bicycle on the road' \\
    --target bicycle --layers d0,d4,s0,s10,s19

  # All 25 layers, all 50 timesteps (slow and potentially several hundred MB):
  uv run python flux2_klein_attention.py \\
    --image scene.png --prompt 'Add a red bicycle on the road' \\
    --target bicycle --layers all --steps 50

Dependencies: torch, diffusers (recent FLUX.2 support), transformers,
accelerate, matplotlib, pillow, numpy, safetensors.
"""
import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import Flux2KleinPipeline


def rotate(x, freqs):
    """FLUX.2 interleaved RoPE (same multiplication order as Diffusers)."""
    cos, sin = freqs
    x_rot = torch.stack((-x[..., 1::2], x[..., 0::2]), dim=-1).flatten(-2)
    return (x.float() * cos[None, :, None, :] +
            x_rot.float() * sin[None, :, None, :]).to(x.dtype)


def target_token_indices(tokenizer, prompt, target, max_length):
    """Find target phrase subword indices in the model's actual chat template."""
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    pos_in_prompt = prompt.casefold().find(target.casefold())
    if pos_in_prompt < 0:
        raise ValueError(f"Target phrase {target!r} not found in prompt")
    prompt_start = formatted.find(prompt)
    if prompt_start < 0:
        raise ValueError("Prompt text was not found in Qwen chat template")
    start = prompt_start + pos_in_prompt
    end = start + len(target)
    encoded = tokenizer(formatted, padding="max_length", truncation=True,
                        max_length=max_length, return_offsets_mapping=True)
    ids = [i for i, (a, b) in enumerate(encoded["offset_mapping"])
           if b > a and a < end and b > start]
    if not ids:
        raise ValueError("Target token indices not found (maybe truncated)")
    pieces = tokenizer.convert_ids_to_tokens([encoded["input_ids"][i] for i in ids])
    print(f"Target tokens: {ids} -> {pieces}")
    return ids


def parse_layers(spec, transformer):
    names = ([f"d{i}" for i in range(len(transformer.transformer_blocks))] +
             [f"s{i}" for i in range(len(transformer.single_transformer_blocks))])
    chosen = names if spec.strip().lower() == "all" else [s.strip() for s in spec.split(",")]
    invalid = set(chosen) - set(names)
    if invalid:
        raise ValueError(f"Invalid layers {sorted(invalid)}. Valid: {names}")
    return list(dict.fromkeys(chosen))


def parse_steps(spec, num_steps):
    if spec.lower() == "all":
        return list(range(num_steps))
    if spec.lower() == "auto":  # old behavior retained if explicitly requested
        return sorted({0, num_steps // 2, num_steps - 1})
    steps = sorted({int(s.strip()) for s in spec.split(",")})
    if any(s < 0 or s >= num_steps for s in steps):
        raise ValueError(f"Step index must be 0..{num_steps - 1}")
    return steps


def resize_long_side(image, max_long_side):
    """Resize proportionally so long side <= max_long_side. No crop."""
    if image is None or max_long_side is None or max_long_side <= 0:
        return image
    w, h = image.size
    long_side = max(w, h)
    if long_side <= max_long_side:
        return image
    scale = max_long_side / float(long_side)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def generation_grid(pipe, source):
    """Match Flux2KleinPipeline's image-resolution decision for the preprocessed input."""
    if source is None:
        return None
    # The pipeline reduces large reference images to <= 1 MP before generating.
    reference = (pipe.image_processor._resize_to_target_area(source, 1024 * 1024)
                 if source.width * source.height > 1024 * 1024 else source)
    alignment = pipe.vae_scale_factor * 2
    working_w = (reference.width // alignment) * alignment
    working_h = (reference.height // alignment) * alignment
    if working_w == 0 or working_h == 0:
        raise ValueError(f"Input image too small for FLUX alignment {alignment}")
    return (working_h // alignment, working_w // alignment), (working_w, working_h)


class AttentionCapture:
    """Non-mutating hooks. Save one probability map per head/layer/denoising step."""

    def __init__(self, text_len, target_ids, grid, steps, query_chunk=32):
        self.text_len = text_len
        self.target_ids = target_ids
        self.grid = grid
        self.gen_tokens = grid[0] * grid[1]
        self.steps = set(steps)
        self.step = 0
        self.query_chunk = query_chunk
        self.maps = {}  # (layer, 0-based step) -> [heads, gh, gw] CPU float16
        self.handles = []

    @torch.no_grad()
    def hook(self, name, attn, args, kwargs):
        marker = (name, self.step)
        # With guidance_scale > 1, Base makes cond forward first and uncond
        # second. Never overwrite cond maps with the uncond pass!
        if self.step not in self.steps or marker in self.maps:
            return
        x = kwargs.get("hidden_states", args[0] if args else None)
        if x is None:
            raise RuntimeError("Missing attention hidden_states; incompatible Diffusers version")
        rope = kwargs.get("image_rotary_emb")
        if rope is None:
            raise RuntimeError("Missing FLUX.2 rotary embeddings")
        n, text_len, dim = self.gen_tokens, self.text_len, attn.head_dim
        if name.startswith("d"):
            ctx = kwargs.get("encoder_hidden_states")
            if ctx is None or ctx.shape[1] != text_len:
                raise RuntimeError("Unexpected double-stream text layout")
            if x.shape[1] < n:
                raise RuntimeError("Not enough generated-image tokens")
            if getattr(attn, "fused_projections", False):
                raise RuntimeError("Unfuse double-stream projections to extract attention")
            # Double stream order: [text, generated image, optional reference].
            q = attn.norm_q(attn.to_q(x[:, :n]).unflatten(-1, (-1, dim)))
            kt = attn.norm_added_k(attn.add_k_proj(ctx).unflatten(-1, (-1, dim)))
            ki = attn.norm_k(attn.to_k(x).unflatten(-1, (-1, dim)))
            k = torch.cat([kt, ki], dim=1)
        else:
            # Single stream order: [text, generated image, optional reference].
            if x.shape[1] < text_len + n:
                raise RuntimeError("Not enough single-stream generated-image tokens")
            linear = attn.to_qkv_mlp_proj
            d = attn.inner_dim
            b = linear.bias
            q = F.linear(x[:, text_len:text_len + n], linear.weight[:d],
                         None if b is None else b[:d])
            k = F.linear(x, linear.weight[d:2*d],
                         None if b is None else b[d:2*d])
            q = attn.norm_q(q.unflatten(-1, (-1, dim)))
            k = attn.norm_k(k.unflatten(-1, (-1, dim)))

        # Image Q positions start immediately after the text tokens.
        q_rope = tuple(a[text_len:text_len + n] for a in rope)
        q = rotate(q, q_rope)
        k = rotate(k, rope)
        q = q[0].permute(1, 0, 2).float()  # [heads, generated patches, dim]
        k = k[0].permute(1, 2, 0).float()  # [heads, dim, ALL keys]
        masses = []
        for chunk in q.split(self.query_chunk, dim=1):
            scores = (chunk @ k) * (dim ** -0.5)
            # Softmax over every text/image key (correct denominator).
            denom = scores.logsumexp(dim=-1)
            target_scores = scores[..., self.target_ids].logsumexp(dim=-1)
            masses.append(torch.exp(target_scores - denom))
        result = torch.cat(masses, dim=1).reshape(q.shape[0], *self.grid)
        self.maps[marker] = result.cpu().to(torch.float16).clone()
        print(f"Captured {name:>3} step {self.step + 1:>2} | {tuple(result.shape)}")

    def install(self, transformer, layers):
        for name in layers:
            attn = (transformer.transformer_blocks[int(name[1:])].attn
                    if name.startswith("d") else
                    transformer.single_transformer_blocks[int(name[1:])].attn)
            def callback(mod, args, kwargs, layer=name):
                self.hook(layer, mod, args, kwargs)
            self.handles.append(attn.register_forward_pre_hook(callback, with_kwargs=True))

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def range_for(maps):
    """One shared display range across a timelapse for meaningful comparison."""
    arrays = [np.asarray(m, dtype=np.float32).ravel() for m in maps]
    z = np.concatenate(arrays)
    low, high = np.percentile(z, [1, 99.5])
    if high <= low:
        high = low + 1e-8
    return float(low), float(high)


def plot_timeline(image, mean_maps, steps, dest, title, columns=10):
    """10 columns x ceil(steps/10) rows, overlay each step on final image."""
    columns = min(columns, max(1, len(steps)))
    rows = math.ceil(len(steps) / columns)
    aspect = image.width / image.height
    tile_w = 2.2
    tile_h = min(3.5, max(1.35, tile_w / max(0.35, aspect)))
    fig, axes = plt.subplots(rows, columns,
                             figsize=(tile_w * columns, (tile_h + .25) * rows),
                             squeeze=False)
    valid_maps = [mean_maps[step] for step in steps if step in mean_maps]
    low, high = range_for(valid_maps) if valid_maps else (0, 1)
    base = np.asarray(image)
    for ax in axes.flat:
        ax.axis("off")
    for i, step in enumerate(steps):
        ax = axes.flat[i]
        ax.imshow(base)
        if step in mean_maps:
            ax.imshow(mean_maps[step], cmap="turbo", vmin=low, vmax=high,
                      extent=(0, image.width, image.height, 0),
                      alpha=0.65, interpolation="bilinear")
        ax.set_title(f"Step {step + 1}", fontsize=10)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .985))
    fig.savefig(dest, dpi=130)
    plt.close(fig)


def plot_heads(image, tensor, layer, step, dest):
    count = tensor.shape[0]
    columns = 6
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows), squeeze=False)
    arr = tensor.float().numpy()
    lo, hi = range_for(arr)
    for ax in axes.flat:
        ax.axis("off")
    for i in range(count):
        ax = axes.flat[i]
        ax.imshow(image)
        ax.imshow(arr[i], cmap="turbo", vmin=lo, vmax=hi,
                  extent=(0, image.width, image.height, 0), alpha=.65)
        ax.set_title(f"Head {i}", fontsize=11)
    fig.suptitle(f"{layer.upper()} | step {step + 1}")
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(dest, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Visualize target image-to-text attention at every denoising step")
    p.add_argument("--model", default="black-forest-labs/FLUX.2-klein-base-4B")
    p.add_argument("--prompt", required=True)
    p.add_argument("--target", required=True, help="Exact phrase within prompt")
    p.add_argument("--image", help="Original source image")
    p.add_argument("--max-long-side", type=int, default=512,
                   help="Resize input proportionally so its long side is at most this value; 0 disables")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--capture-steps", default="all", help="all (default), auto, or 0-based comma list")
    p.add_argument("--layers", default="d0,d4,s0,s10,s19", help="0-based d0..d4,s0..s19 or all")
    p.add_argument("--guidance-scale", type=float, default=4.0)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, help="Text-to-image only; omitted for image editing")
    p.add_argument("--height", type=int, help="Text-to-image only; omitted for image editing")
    p.add_argument("--query-chunk", type=int, default=32,
                   help="Queries per exact QK chunk; decrease if memory is low")
    p.add_argument("--output", default="attention_results")
    args = p.parse_args()
    if args.steps < 1 or args.query_chunk < 1:
        p.error("--steps and --query-chunk must be positive")
    if args.image and (args.width is not None or args.height is not None):
        p.error("Do not specify --width/--height with --image: output follows source")
    selected_steps = parse_steps(args.capture_steps, args.steps)
    if not selected_steps:
        p.error("No capture steps selected")

    device = ("cuda" if torch.cuda.is_available() else
              "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    print("Loading", args.model, "on", device)
    pipe = Flux2KleinPipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    layers = parse_layers(args.layers, pipe.transformer)
    token_ids = target_token_indices(pipe.tokenizer, args.prompt, args.target, args.max_seq_len)

    source_original = Image.open(args.image).convert("RGB") if args.image else None
    source = resize_long_side(source_original, args.max_long_side) if source_original is not None else None
    if source is not None:
        grid, working_size = generation_grid(pipe, source)
    else:
        w = args.width or 1024
        h = args.height or 1024
        alignment = pipe.vae_scale_factor * 2
        working_size = ((w // alignment) * alignment, (h // alignment) * alignment)
        grid = (working_size[1] // alignment, working_size[0] // alignment)
        if min(working_size) == 0:
            p.error("Requested text-to-image resolution is too small")
    print(f"Original input size={source_original.size if source_original is not None else None}; "
          f"resized input size={source.size if source is not None else None}; "
          f"model output working size={working_size}; grid={grid}")
    print(f"Capturing {len(layers)} layers x {len(selected_steps)} steps "
          f"(conditional pass only).")
    capture = AttentionCapture(args.max_seq_len, token_ids, grid, selected_steps,
                               query_chunk=args.query_chunk)
    capture.install(pipe.transformer, layers)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    def step_end(_pipe, step, _t, kwargs):
        capture.step = step + 1
        return kwargs

    params = dict(prompt=args.prompt, image=source, num_inference_steps=args.steps,
                  guidance_scale=args.guidance_scale, max_sequence_length=args.max_seq_len,
                  generator=torch.Generator(device="cpu").manual_seed(args.seed),
                  callback_on_step_end=step_end)
    if source is None:
        params.update(width=working_size[0], height=working_size[1])
    try:
        with torch.inference_mode():
            result = pipe(**params).images[0]
    finally:
        capture.close()

    # Keep output at the resized inference size used for attention extraction.
    if source is not None and result.size != source.size:
        result = result.resize(source.size, Image.Resampling.LANCZOS)
    result.save(outdir / "generated.png")

    missing = [(layer, step) for layer in layers for step in selected_steps
               if (layer, step) not in capture.maps]
    if missing:
        raise RuntimeError(f"Failed to capture {len(missing)} maps: {missing[:8]}")

    torch.save({"maps": capture.maps, "target": args.target, "prompt": args.prompt,
                "layers": layers, "steps": selected_steps, "grid": grid,
                "working_size": working_size,
                "original_input_size": source_original.size if source_original is not None else None,
                "resized_input_size": source.size if source is not None else None,
                "max_long_side": args.max_long_side,
                "token_indices": token_ids, "guidance_scale": args.guidance_scale},
               outdir / "attention.pt")

    # Visualize EVERY selected step for EACH selected layer.
    for layer in layers:
        series = {step: capture.maps[layer, step].float().mean(dim=0).numpy()
                  for step in selected_steps}
        path = outdir / f"attention_timesteps_{layer}.png"
        plot_timeline(result, series, selected_steps, path,
                      f"{layer.upper()} | {args.target!r} | mean over heads")
        print("Wrote", path)

    # And an aggregate view across all selected layers for all timesteps.
    combined = {step: torch.stack([capture.maps[layer, step].float().mean(dim=0)
                                   for layer in layers]).mean(dim=0).numpy()
                for step in selected_steps}
    plot_timeline(result, combined, selected_steps, outdir / "attention_overview.png",
                  f"{args.target!r} | mean over {len(layers)} layers and heads")

    example_layer = "s10" if "s10" in layers else layers[-1]
    example_step = selected_steps[len(selected_steps) // 2]
    plot_heads(result, capture.maps[example_layer, example_step],
               example_layer, example_step, outdir / "attention_heads.png")
    print(f"Saved {len(layers) * len(selected_steps)} maps in {outdir.resolve()}")


if __name__ == "__main__":
    main()
