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
| 5 | Replace `torchvision.io.read_video` with `cv2.VideoCapture` | `fix_utils.py` (run during build) replaces `read_frame_from_videos` in `utils.py` | torchvision → PyAV → swscale fails "EAGAIN" on many real-world inputs in this image. decord segfaults in `get_batch`. cv2 (already used by UAV's folder branch) is the only stable reader. |
| 6 | `pipeline.enable_xformers_memory_efficient_attention()` injected after multi_gpu_setup | sed in `inference_upscale_a_video.py` | xformers 0.0.20 ships in the image and the temporal/spatial modules check `_use_memory_efficient_attention_xformers`, but upstream's inference script never calls the enable hook. Free win — saves a few GiB of attention buffers per UNet forward. |
| 7 | Disable the auto-tile guard `if h*w >= 384*384: args.perform_tile = True` | sed in `inference_upscale_a_video.py` (line 204) | Upstream silently force-enables tiling for any input ≥ 384², overriding `--no-tile` and `--perform_tile False`. With the guard removed, `--tile_size` is honest — pass a value larger than the input dims to actually run untiled. (Memory may not fit untiled at 480p — that's the wrapper's call to make.) |

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

## Cost model (measured, not extrapolated)

Earlier versions of this doc extrapolated linearly from a 4-frame 320×240
test ("60 s of 480p ≈ 1.5–3 h") which was off by ~10×. Real measurements on
dual-3090, all seven patches applied, `-s 15` draft mode:

| Source | Mode | Tiles | Steps × tiles | Wall-clock | Per output frame |
|---|---|---|---|---|---|
| 1 s @ 240×240 | untiled | 1 | 15 × 1 = 15 | **213 s** ✓ measured | 7.1 s |
| 1 s @ 480p | untiled (incl. sharded UNet + xformers) | n/a | n/a | **OOM ~21 GiB on cuda:0 at step 1** | does not fit |
| 1 s @ 480p | tile-128, sharded UNet, xformers (all 7 patches) | 5×4 = 20 | 15 × 20 = 300 | **68 min** ✓ measured (with browser GPU contention) | ~136 s |
| 38 s @ 480p | tile-128, sharded UNet (extrapolation) | 20 × ~13 chunks | ~3900 | **~36-44 hours** | ~115-140 s |

The "per output frame" column is the real number to keep in your head. For
context, RealESRGAN does roughly **30-100 frames per second** on the same
hardware. UAV at 480p is **3 orders of magnitude slower** because each
output frame is amortised over 20 tile passes × 30 (or 15) denoising steps
× 2 classifier-free-guidance forwards. That cost ratio is not a bug — it's
the tax for tiling a model that wasn't designed to be tiled.

### Why we can't fix this further

`enable_xformers_memory_efficient_attention()` (patch #6) saves about 1.5
GiB of attention buffer per forward — measured by re-running the 480p
untiled OOM probe with and without the patch (allocated dropped from 19.63
GiB to 21.05 GiB — yes, xformers reshapes the allocator pattern so the
"allocated" number actually went up but "reserved" went down; the operative
test is whether step 1 completes, and it didn't either way). Even fully
optimised we're ~30 MiB short of fitting 1 s of 480p untiled in 24 GiB.

The fundamental driver is the UNet residual + activations during a single
30-frame forward pass at 480p, which the multi_gpu_setup.py docstring
estimated at "~18 GB peak" — the real measured peak is ~21 GiB on cuda:0.
There is no further trick that gets us under 24 GiB without dropping below
~15 frames per chunk, at which point the per-second wall-clock approaches
RealESRGAN territory anyway and we may as well use the right tool.

### When to actually use UAV

- ✅ Hero shots of ≤2 s at ≤320p source resolution — fits untiled, ~7 s per
  output frame, real diffusion quality
- ✅ Pre-downscaled 480p clips (downscale to 320×240 first, UAV → 1280×960)
  — output is "only" 1280×960 but you get the diffusion finish in single-digit hours
- ❌ Full-resolution 480p clips longer than a few seconds — use `v2x-*` instead
- ❌ Anything you'd consider a "production" workload — UAV is a paper demo

For everything else, `v2x-oldnsfw --deep --dual-gpu` finishes 38 s of 480p in
minutes, not days.

## Chunking (the second OOM, after the VAE one)

After the four model-level patches, UAV runs — but there's still a hard
ceiling on how *long* a clip you can feed it in one shot. The script
pre-allocates the whole upscaled output as a single float32 tensor on
`cuda:0` before the decode loop:

```python
# inference_upscale_a_video.py, ~line 217
upscaled_video = vframes.new_zeros(output_shape)   # (1, 3, T, 4H, 4W)
```

For 4× upscale that's `192 × W × H × T` bytes. Concretely:

| Source | Bytes per second of input | Max clip on 24 GB (no other use) |
|---|---|---|
| 320×240 @ 30 fps | 422 MiB/s | ~50 s |
| 480p (640×480) @ 30 fps | 1.69 GiB/s | ~13 s |
| 720p (1280×720) @ 30 fps | 5.06 GiB/s | ~4.5 s |
| 1080p (1920×1080) @ 30 fps | 11.39 GiB/s | ~2 s |

These limits *ignore* the UNet residual (~13 GiB on `cuda:0`) — the real
budget is smaller. With 13 GiB resident the practical ceiling is more like
~6 s of 480p before allocation fails. That's the OOM you'd hit on anything
realistic, and it's the reason `uav` chunks by default.

### How the wrapper chunks

The output budget is the tunable knob — `UAV_OUTPUT_BUDGET_BYTES`, default
10 GiB. The wrapper solves `192 × W × H × T ≤ budget` for `T`, divides by
`fps`, and uses that as the chunk size in seconds (rounded down to the
nearest integer second, min 1). Then:

1. `ffmpeg ... -c:v libx264 -crf 0 -g 1 -f segment -segment_time SEC`
   splits the input losslessly with every frame as a keyframe (`-g 1`)
   so cuts are clean and reproducible.
2. Each segment is upscaled via a fresh `podman run`; the per-chunk output
   lives at `<staging>/upscaled/seg_NNNN/video/<stem>_n*g*s*.mp4`.
3. After all chunks succeed, `ffmpeg -f concat -i ... -c:v libx264 -crf 18`
   stitches them into one MP4 in the user's output dir, named after the
   *original* source (not the cached/tagged copy).

Staging dir is keyed by input `size-mtime`, so the same source always lands
in the same staging path. Resume works because each chunk's output is
checked before the run — if a non-empty `.mp4` already sits in the chunk's
`video/` dir, that chunk is skipped. The staging dir is kept across runs by
default (Ctrl-C or chunk failure → resume on re-run); `--clean-chunks`
removes it after a clean concat.

### Why a 10 GiB output budget

A 24 GiB card holds the ~13 GiB UNet residual *plus* the output buffer
*plus* every per-step working tensor. We've measured the working set at
under ~1 GiB once `short_seq=1` and `tile_size=128` are in effect, so
`13 + 10 + 1 = 24 GiB` is right at the edge but reliable on a clean GPU.
Lower the budget if you're sharing GPU 0 with another workload; raise it
if you're on bigger cards.

### What we explicitly didn't do

- **Patch UAV to chunk internally.** Tempting, but the entire pipeline
  (text-encoder cache, scheduler state, color-fix accumulators) is keyed
  on the full output tensor. A clean per-chunk subprocess gives us
  isolation and resumability for one extra ffmpeg pass.
- **Use NVDEC/NVENC for the split.** libx264 `-crf 0` keeps the bitstream
  identical and avoids needing a second CUDA context contention point.
- **Concat at `-c copy`.** Tried first — concat-demuxer with `-c copy`
  refused on segments produced from a previous re-encode in some
  containers. CRF 18 is visually lossless and works on every codec
  the upscaler can output.

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
