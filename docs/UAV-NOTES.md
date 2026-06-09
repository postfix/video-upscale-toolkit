# UAV on 24 GB cards — engineering notes

Upscale-A-Video ([sczhou/Upscale-A-Video][repo]) is a diffusion-based video
super-resolution model. Its README quietly assumes ≥ 48 GB VRAM and a single
card; on a 24 GB 3090 it OOMs out of the box. Even the project's own issue
tracker says "4090 (24 GB) couldn't finish the demo video" (#23, #25, #28).

This document is the story of the four patches in our Dockerfile and why
each is necessary. Keep it around — without it, the next person who looks
at the patches will think "why is the VAE specifically in bf16?" and revert
something that took hours to find.

[repo]: https://github.com/sczhou/Upscale-A-Video

## TL;DR

Out of the box: **OOM at the first VAE decode**, tries to allocate 65.92 GiB.
After all four patches: **works on dual 24 GB cards**, valid output, ~115 s
for 4 frames at 320×240.

## The patches

| # | What | Where | Why |
|---|------|-------|-----|
| 1 | Multi-GPU split: UNet on `cuda:0`, VAE on `cuda:1` | `multi_gpu_setup.py` + sed at line 131 of `inference_upscale_a_video.py` | UNet residual (~13 GB) + VAE decode (~15 GB) > 24 GB on one card. Two cards = each component has 24 GB to itself. |
| 2 | VAE → BF16 (not FP16) | sed at lines 107/111 of `inference_upscale_a_video.py` | The script does `pipeline.unet = pipeline.unet.half()` but **never halves the VAE** — it stays FP32 from `from_config`. We can't use plain FP16: the VAE genuinely overflows ("invalid value encountered in cast" → all-black frames). **BF16 has the same exponent range as FP32** but half the memory of FP16 — exactly what we need. |
| 3 | Skip the inline `self.vae.to(dtype=torch.float32)` | sed in `pipeline_upscale_a_video.py` | Inside the pipeline's `__call__`, the original code force-promotes the VAE to FP32 right before decode (legacy of the no-bf16-Ampere days). With patch #2 the VAE is already in bf16 with safe range — this promotion would defeat #2 and bring back the OOM. |
| 4 | `short_seq = 1` (was 3) | sed in `pipeline_upscale_a_video.py` | The video VAE decodes `short_seq` frames per chunk. Temporal cross-attention scales **quadratically** with `short_seq`: 3 → 1 cuts the attention buffer 9× (~13 GB → ~1.5 GB). The cost is 3× more decode-loop iterations but each is tiny and fast. |

## The chase, in order

This is roughly the sequence of failures and partial fixes I went through.
Future me: don't re-derive these from scratch.

1. **Baseline** — single card, default args. OOM trying to allocate **65.92 GiB**
   at "Decoding: 0/2". Same number with or without `--use_video_vae`.
2. **`--perform_tile --tile_size 256`** — also 65.92 GiB. Tiling doesn't help
   when the whole image fits in one tile.
3. **`--tile_size 128`** — OOM drops to 15.19 GiB allocation. But residual
   was 19 GB by the time decode starts → still doesn't fit. (Useful data point:
   the buffer scales with `tile_size²`.)
4. **`enable_model_cpu_offload(gpu_id=0)`** — cut residual to 5 GB at
   `tile_size=256`, but at `tile_size=128` the residual climbed back to
   18 GB because the offload hooks don't fire between U-Net steps in the
   denoising loop. Diffusers' offload mechanism doesn't track UAV's custom
   `VideoUpscalePipeline` cleanly.
5. **`enable_sequential_cpu_offload`** — even more aggressive, saved
   another 2 GB. Still 17 GB residual + 15 GB allocation = doesn't fit.
6. **Two-GPU split** (patch #1) — finally moves the bottleneck to `cuda:1`.
   Same OOM numbers, just on the other card. Confirms VAE alone needs > 24 GB.
7. **Track the 14 GB residual on cuda:1** — turns out the VAE itself is in
   FP32 because the inference script never calls `.half()` on it (the UNet
   does, but the VAE replacement after `from_config` doesn't carry dtype).
   → patch #2 (initially as `.half()`, then bf16 in step 9).
8. **VAE `.half()`** — residual drops to 14.75 GB and allocation to 13.50 GB.
   Output is **all black**: YMIN = YAVG = YMAX = 16 (YUV black). FP16 VAE
   overflows, NaNs propagate, `.astype(np.uint8)` quietly produces zeros.
9. **VAE `.to(dtype=torch.bfloat16)`** instead — same memory as fp16 but
   full fp32 exponent range. **Valid output.** This is the patch.
10. **`short_seq = 1`** (patch #4) — fits everything inside `cuda:1`'s
    budget with margin. Also fixes the "tensor 768 must match 512" shape
    bug that appeared at smaller tile sizes (it was a downstream artifact
    of the failed decode, not a real bug).

## What we explicitly didn't do

- **Smaller `tile_size` (64 or 96)** — hits a tile-composition shape bug in
  UAV when `image_width % tile_size != 0`. Not worth fighting; 128 works.
- **`device_map="auto"` via accelerate** — diffusers 0.16.0 (what UAV pins)
  has limited support; the auto-placement landed odd things on CPU. Our
  hand-rolled split into two GPUs is more predictable.
- **`enable_vae_slicing()`** — works on diffusers' stock `AutoencoderKL`
  but UAV's `AutoencoderKLVideo` subclass doesn't implement it.
- **Replacing the VAE with a stock SDXL VAE** — possible, but the upstream
  VAE was trained jointly with UAV's UNet; substituting it would change
  output quality unpredictably.

## Cost model

After all patches, on a dual-3090 host:

| Source | Tile-128 passes | Approx wall-clock |
|---|---|---|
| 4 frames @ 320×240 | ~6 tile rounds | ~2 min |
| 5 s @ 480p (24 fps = 120 frames) | ~30× the 4-frame test | 8–15 min |
| 60 s @ 480p (1440 frames) | ~360× | 1.5–3 hours |

Throughput is bottlenecked by `tile_size=128` and 30 inference steps. For
"draft quality" preview you can drop `-s` to 15 (~50 % faster).

## Re-running these patches against a future UAV release

If upstream UAV gets updated, the sed patterns in the Dockerfile may need
adjusting:

- Patch #1 anchors on `^    pipeline = pipeline.to(UAV_device)$` — check
  line ~131 of `inference_upscale_a_video.py`.
- Patch #2 anchors on `^        pipeline\.vae\.load_state_dict(.*)$` — check
  both VAE branches (lines ~107 and ~111).
- Patch #3 anchors on `^        self\.vae\.to(dtype=torch\.float32)$` —
  check around line 668 of `pipeline_upscale_a_video.py`.
- Patch #4 anchors on `^        short_seq = 3$` — same file, around line 684.

Each sed has a `grep -q` guard that fails the build if the anchor moved.
That's intentional: if a patch silently fails to apply, you get black
frames or OOMs again, and you'd think the model itself broke.
