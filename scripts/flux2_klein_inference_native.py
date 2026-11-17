"""FLUX.2 Klein 4B Base image editing without attention hooks.

Downscale the input proportionally if its long side exceeds 512 pixels.
No crop or stretching. Save output at the resulting inference input size.
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from diffusers import Flux2KleinPipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True)
    p.add_argument('--prompt', required=True)
    p.add_argument('--output', default='output.png')
    p.add_argument('--model', default='black-forest-labs/FLUX.2-klein-base-4B')
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--guidance-scale', type=float, default=4.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-long-side', type=int, default=512,
                   help='Maximum long side for input image (0 disables resize)')
    args = p.parse_args()

    image = Image.open(args.image).convert('RGB')
    original_size = image.size  # (width, height)
    if args.max_long_side > 0 and max(image.size) > args.max_long_side:
        scale = args.max_long_side / max(image.size)
        size = (max(1, round(image.width * scale)),
                max(1, round(image.height * scale)))
        image = image.resize(size, Image.Resampling.LANCZOS)
    inference_size = image.size

    device = 'cuda' if torch.cuda.is_available() else (
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )
    dtype = torch.bfloat16 if device != 'cpu' else torch.float32
    pipe = Flux2KleinPipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipe.to(device)

    generator = torch.Generator(device='cpu').manual_seed(args.seed)
    with torch.inference_mode():
        result = pipe(
            image=image,                 # long side <= 512 by default
            prompt=args.prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]

    model_size = result.size
    if result.size != inference_size:
        result = result.resize(inference_size, Image.Resampling.LANCZOS)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.save(path)
    print(f'Original: {original_size}; resized input: {inference_size}; pipeline: {model_size}; saved: {result.size}')
    print(f'Saved {path.resolve()}')


if __name__ == '__main__':
    main()
