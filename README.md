# video-upscale-toolkit

Podman-based video upscaling stack for Debian-class hosts with **two 24 GB
NVIDIA GPUs** (e.g. dual RTX 3090 / 4090). Two engines, one wrapper style:

- **Video2X** (`ghcr.io/k4yt3x/video2x`) — fast NCNN/Vulkan upscaling
  (Real-ESRGAN, Real-CUGAN, RIFE, Anime4K via libplacebo). Real-time-ish on
  a 3090. Used for the bulk of work.
- **Upscale-A-Video** (custom-built) — diffusion-based super-resolution
  ([sczhou/Upscale-A-Video][uav]). 10–30× slower but much better on heavily
  degraded sources. Patched to fit in dual-24 GB.

[uav]: https://github.com/sczhou/Upscale-A-Video

## What's in here

```
bin/                          # wrappers to drop into ~/.local/bin/
  video2x                     # main Video2X podman wrapper (--dual-gpu, --model, --profile)
  v2x-home                    # live-action preset (NMKD-Siax)
  v2x-anime                   # anime video (realesr-animevideov3)
  v2x-anime-hd                # anime stills (HFA2K)
  v2x-nsfw                    # adult content (Remacri + x265 10-bit)
  v2x-oldnsfw                 # VHS-era pipeline (ffmpeg pre-clean + Siax → Remacri)
  v2x-install-models          # downloads & profiles 6 community NCNN models
  uav                         # UAV podman wrapper (multi-GPU sharded)
  uav-install-models          # fetches UAV pretrained weights (~10 GB)

containers/uav/               # local container build for UAV
  Dockerfile                  # PyTorch 2.0.1 + CUDA 11.7 + UAV deps + 4 memory patches
  multi_gpu_setup.py          # runtime hook: UNet on cuda:0, VAE on cuda:1

docs/
  CHEATSHEET.md               # recipes per content type, dual-GPU usage
  UAV-NOTES.md                # the engineering rabbit hole — why each UAV patch exists
```

## Host requirements

- Debian 13 / Ubuntu 24.04 (any glibc ≥ 2.36 should work)
- Podman ≥ 4.0 (this build verified on 5.4.2)
- NVIDIA driver ≥ 535 with two CUDA-capable GPUs of equal class
- `nvidia-container-toolkit` ≥ 1.19 with a CDI spec at `/etc/cdi/nvidia.yaml`
- `ffmpeg` ≥ 6 on the host (used by `v2x-oldnsfw` and `video2x --dual-gpu`)

CDI spec is generated once:

```bash
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
nvidia-ctk cdi list   # should show nvidia.com/gpu=0, =1, =all
```

## Install

See [`QUICKSTART.md`](QUICKSTART.md) for the 15-minute setup. Short version:

```bash
git clone https://github.com/postfix/video-upscale-toolkit.git
cd video-upscale-toolkit
./install.sh                          # copies bin/* → ~/.local/bin/

# Video2X side (downloads ~270 MB of community ESRGAN models)
v2x-install-models

# Upscale-A-Video side (builds ~10 GB image, fetches ~10 GB weights)
podman build -t localhost/uav:latest containers/uav/
uav-install-models                    # may need manual zip download (see docs)
```

`~/.local/bin/` must be on your PATH (it usually is on modern shells).

## Quick start

```bash
# Live-action 4× upscale, x264 CRF 18:
v2x-home in.mp4 out.mp4

# Use both 3090s on one file:
v2x-nsfw big.mkv big-up.mkv --dual-gpu

# Old VHS-era footage: ffmpeg pre-clean → NMKD-Siax → Remacri texture pass:
v2x-oldnsfw vhs.mp4 restored.mkv --deep --dual-gpu

# Diffusion-quality on a 30 s hero clip:
uav clip.mp4 ./out --prompt "vintage 1980s home video, film grain"
```

Full recipes are in [`docs/CHEATSHEET.md`](docs/CHEATSHEET.md).
Diffusion prompts for `uav` are in [`docs/PROMPTS.md`](docs/PROMPTS.md).
Runtime errors and their fixes are in [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md).
The story behind the UAV patches is in [`docs/UAV-NOTES.md`](docs/UAV-NOTES.md).

## Why these tools, why this hardware

Built for Debian 13 + 2× RTX 3090. The whole stack assumes:

- Two GPUs with **24 GB each** — UAV's multi-GPU patch shards UNet onto
  `cuda:0` and VAE onto `cuda:1`. On a single 24 GB card UAV silently
  produces black frames (fp16 VAE overflow) or OOMs at decode.
- **Podman** (not Docker). Both wrappers use CDI-style device passthrough
  (`--device nvidia.com/gpu=all`). On Docker you'd use `--gpus all` instead.
- Rootless containers; nothing here needs `sudo` after the one-time CDI
  setup.

Everything else (Real-ESRGAN models, codecs, presets) is configurable.

## License

MIT — use, modify, redistribute. See `LICENSE`.
