# Home-video restoration cheat sheet (Video2X + dual RTX 3090)

Home camcorder / phone footage is **live-action**, not anime, so pick the
live-action models. Anime models (`realesr-animevideov3`, RealCUGAN, Anime4K
shaders) will smear faces and skin texture — avoid them.

---

## 🐢 Hero-clip upscaling — `uav` (Upscale-A-Video)

For 5–60 second clips where you want maximum quality. UAV is a diffusion-based
video super-resolution model (PyTorch + CUDA). On **dual 24 GB cards** it runs
with the patches baked into our container (multi-GPU split + BF16 VAE +
short_seq=1) — about ~30 s/tile-of-128px on a 3090 pair.

```bash
# one-time setup
podman build -t localhost/uav:latest ~/.local/share/uav/build      # ~10 GB image
uav-install-models                                                  # ~10 GB models from Google Drive

# usage
uav clip.mp4 ./out                                                  # multi-GPU defaults, tile=128, bf16
uav clip.mp4 ./out --prompt "vintage 1980s home video, film grain"  # guide the diffusion
uav clip.mp4 ./out -- -n 100 -g 5 -s 25                             # gentler / faster
uav --shell                                                         # debug inside the container
```

### Patches baked into the container

UAV out of the box does **not** fit in 24 GB. Our Dockerfile applies four
patches; you don't need to think about them but knowing they're there helps
when debugging:

| Patch | What it does | Why |
|---|---|---|
| `multi_gpu_setup.py` | UNet+text_encoder → `cuda:0`, VAE → `cuda:1` | Splits the ~32 GB peak across two cards |
| VAE → BF16 | Add `.to(dtype=torch.bfloat16)` after the VAE load | UAV's "fp16 overflows" comment is true; bf16 has fp32 range + fp16 memory |
| Skip FP32 cast | Comment out `self.vae.to(dtype=torch.float32)` in pipeline | Original cast doubled VAE memory; with bf16 no longer needed |
| `short_seq = 1` | One frame per VAE decode chunk instead of 3 | Temporal cross-attention is T² — cuts that buffer 9× |

### Reality check

- **5 s of 480p source ≈ 8–15 min** on dual 3090s (tile_size=128, 30 steps)
- **60 s of 480p ≈ 1.5–3 h**
- Quality is genuinely better than Real-ESRGAN for soft/old footage, but
  diffusion **can hallucinate detail** — keep the source if you want truth
- The bf16 VAE drops a tiny bit of decode fidelity vs the original fp32 cast;
  not visible side-by-side on most content

When the per-second cost matters, `v2x-oldnsfw --deep --dual-gpu` is still
~10× faster and often "good enough". Reach for `uav` for short prized clips.

---

## ⚡ Use both 3090s on one file — `--dual-gpu`

A single Video2X process is **always single-GPU** (ncnn-vulkan limitation —
the second 3090 sits idle). Adding `--dual-gpu` to any wrapper splits the
input at I-frame boundaries, runs the two halves in parallel pinned to
`-d 0` and `-d 1`, and concatenates losslessly. Wall-clock ≈ ½.

```bash
v2x-nsfw  big.mp4 big-up.mkv --dual-gpu
v2x-home  vacation.mp4 vacation-up.mp4 --dual-gpu
v2x-anime episode.mkv episode-4k.mkv --dual-gpu
v2x-oldnsfw vhs.mp4 restored.mkv --deep --dual-gpu     # both passes use both cards
video2x   in.mp4 out.mp4 --model ultrasharp --dual-gpu
```

Requirements: host `ffmpeg` + `ffprobe` (you have them). Per-call overhead
is ~5 s for split + concat — only worth it on clips longer than ~30 s.

## 0. One-shot presets (the easy path)

Run **once** to download community models (~270 MB):

```bash
v2x-install-models
```

Then:

| Preset | Use for | Model (via profile) | Codec |
|---|---|---|---|
| `v2x-home in out`     | home / VHS / phone / live-action  | **4x_NMKD-Siax_200k** (compressed-source denoiser) | x264 CRF 18 yuv420p |
| `v2x-anime in out`    | anime / cartoon video             | realesr-animevideov3 (temporal-aware) | x264 CRF 17 yuv420p |
| `v2x-anime-hd in out` | anime stills / recent HD anime    | **4xHFA2k** (sharper, modern) | x264 CRF 17 yuv420p |
| `v2x-nsfw in out`     | adult / skin-tone-heavy           | **foolhardy Remacri** (preserves skin) | x265 CRF 20 yuv420p10le |
| `v2x-oldnsfw in out`  | VHS-era / heavily-compressed adult footage | ffmpeg pre-clean → NMKD-Siax → x265 (`--deep` adds Remacri texture pass) | x265 CRF 20 yuv420p10le |

Every preset accepts `--dual-gpu` to use both 3090s (see top of doc).

### General-purpose model picker — `video2x --model NAME`

For everything else, swap the active model on the main wrapper:
```bash
video2x in.mp4 out.mp4 --model ultrasharp       # very sharp, line art / edges
video2x in.mp4 out.mp4 --model high-fidelity    # preserves detail in clean sources
video2x in.mp4 out.mp4 --model lsdir-plus       # modern general-purpose default
```
Available `--model` values: `ultrasharp`, `high-fidelity`, `lsdir-plus`,
`nmkd-siax`, `remacri`, `hfa2k`. Each maps to a profile dir built by
`v2x-install-models`.

All presets are 4× upscale (only scale ESRGAN-plus-class weights ship in). Override per call:
```bash
v2x-nsfw in.mp4 out.mkv -d 1                # second 3090
v2x-home in.mp4 out.mp4 -- -e crf=16        # tighter CRF
```

### How profiles work (advanced)

Video2X's CLI rejects unknown `--realesrgan-model` names. Workaround: each
**profile** is a model directory where the `realesrgan-plus` slot is swapped
with a community model. `v2x-install-models` builds four profiles:
`default`, `live-action`, `nsfw`, `anime-stills`. The wrapper bind-mounts the
requested one over `/usr/share/video2x/models/realesrgan` inside the container.

Use a profile manually:
```bash
video2x in.mp4 out.mp4 --profile live-action --realesrgan-model realesrgan-plus -s 4
```
or pin per shell:
```bash
export VIDEO2X_PROFILE=nsfw
```

Other community models downloaded but **not wired to a preset** (use via profile
swap by editing `v2x-install-models` and re-running):
`ultrasharp`, `high-fidelity`, `lsdir-plus`.

Each accepts any extra `video2x` flag (and last-wins, so user flags override the preset):

```bash
v2x-home  vacation.mp4 vacation-2x.mp4               # default: 2× live-action
v2x-anime ep01.mkv     ep01-4k.mkv      -d 1         # use 2nd 3090
v2x-nsfw  clip.mp4     clip-up.mkv      -s 4         # override scale to 4×
v2x-home  clip.mp4     clip-up.mp4      -- -e crf=16 # extra raw video2x flags
```

Sections 1+ show what each preset expands to and how to hand-craft variants.

---

## 1. The default recipe — 4× upscale, denoise, clean encode

```bash
video2x in.mp4 out.mp4 \
  --realesrgan-model realesrgan-plus -s 4 \
  -c libx264 --pix-fmt yuv420p \
  -- -e crf=18 -e preset=slow -e tune=film
```
Best general starting point for VHS rips, old phone clips, or compressed 480p/720p.
RealESRGAN-plus is the live-action model; it also cleans compression noise.
(Scale is fixed at 4 — the model has no 2×/3× weights.)

## 2. Big upscale — 480p → 4K-ish

```bash
video2x in.mp4 out.mp4 \
  --realesrgan-model realesrgan-plus -s 4 \
  -c libx264 --pix-fmt yuv420p \
  -- -e crf=17 -e preset=veryslow -e tune=film
```
4× is the maximum useful scale for live-action; beyond that you hallucinate
detail that wasn't there.

## 3. Smooth motion — 30 fps → 60 fps via RIFE

```bash
video2x in.mp4 out.mp4 -m 2 --rife-model rife-v4.26
```
`-m 2` doubles fps, `-m 4` quadruples it. `rife-v4.26` is the newest general model.

## 4. The full restore — upscale **then** interpolate (two passes)

```bash
video2x in.mp4 tmp-up.mp4 --realesrgan-model realesrgan-plus -s 2 \
  -c libx264 --pix-fmt yuv420p -- -e crf=17 -e preset=slow

video2x tmp-up.mp4 out.mp4 -m 2 --rife-model rife-v4.26
```
Upscale-first preserves more interpolation accuracy than the reverse order.

## 5. Faster encode — H.265 on the GPU (NVENC)

```bash
video2x in.mp4 out.mkv \
  --realesrgan-model realesrgan-plus -s 2 \
  -c hevc_nvenc --pix-fmt yuv420p \
  -- -e preset=p7 -e tune=hq -e rc=vbr -e cq=20 -e b:v=0
```
Encoder bottleneck disappears — useful for long clips. Quality is slightly below
`libx264 crf 18` at the same file size; for archival keep CPU x264, for
preview/quick wins use NVENC.

## 6. Both 3090s, two clips at once

```bash
video2x clipA.mp4 clipA-out.mp4 -d 0 &
video2x clipB.mp4 clipB-out.mp4 -d 1 &
wait
```
Single Video2X run is single-GPU. Parallelise across files, not within.

## 7. Batch a whole folder

```bash
mkdir -p out
i=0
for f in raw/*.mp4; do
  gpu=$(( i % 2 ))
  video2x "$f" "out/$(basename "$f" .mp4)-4x.mp4" \
    --realesrgan-model realesrgan-plus -s 4 -d "$gpu" \
    -c libx264 --pix-fmt yuv420p \
    -- -e crf=18 -e preset=slow -e tune=film &
  (( ++i % 2 == 0 )) && wait        # 2 jobs at a time, one per GPU
done
wait
```

---

## Pre-processing tough sources

Run these with the *host* ffmpeg first, then feed the result to `video2x`.

| Problem | One-liner |
|---|---|
| Interlaced DV / VHS capture | `ffmpeg -i in.avi -vf "yadif=1" -c:v ffv1 -c:a copy clean.mkv` |
| Heavy chroma noise | `ffmpeg -i in.mp4 -vf "hqdn3d=4:3:6:4.5" -c:v ffv1 -c:a copy clean.mkv` |
| Wrong colour cast | `ffmpeg -i in.mp4 -vf "colorbalance=rs=.1:gs=-.05" ... ` |
| Shaky handheld | `ffmpeg -i in.mp4 -vf vidstabdetect=shakiness=8 -f null -` then `vidstabtransform` |

Use `ffv1`/`prores` as intermediate codec — lossless, so Video2X has clean input.
Don't pre-upscale; let Real-ESRGAN do it.

---

## Quality / time knobs

| Knob | Effect |
|---|---|
| `-s 2` vs `-s 4` | 4× is ~4× slower and not always sharper for live-action |
| `--realesrgan-model realesrgan-plus` | **live-action** (use this for home video) |
| `--realesrgan-model realesrgan-plus-anime` | anime only — skip for home video |
| `--realesrgan-model realesr-animevideov3` | anime only — skip for home video |
| `-c libx264 -e preset=veryslow -e crf=17` | archival CPU encode, ~6× slower than `preset=slow` |
| `-c hevc_nvenc -e preset=p7 -e cq=20` | GPU encode, ~10× faster, slightly larger files |
| `-m 2` / `-m 4` | RIFE multiplier; doubles or quadruples fps |
| `-d 0` / `-d 1` | pin to a specific 3090 |
| `-b` | benchmark: discard output, just print FPS |

---

## Quick sanity check before a long run

```bash
ffmpeg -i in.mp4 -ss 30 -t 10 -c copy sample.mp4   # 10-sec sample at 0:30
video2x sample.mp4 sample-out.mp4 --realesrgan-model realesrgan-plus -s 4
```
Eyeball the 10-second output before kicking off the multi-hour full pass.

## Watch GPU load while running

```bash
watch -n 1 nvidia-smi          # both 3090s should sit near 100% util
```
