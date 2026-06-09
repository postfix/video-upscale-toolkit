# Quickstart — 15 minutes from `git clone` to first upscale

Fast path for a fresh Debian 13 / Ubuntu 24.04 host with two NVIDIA 24 GB
GPUs (e.g. 2× RTX 3090). See [`README.md`](README.md) for background and
[`docs/CHEATSHEET.md`](docs/CHEATSHEET.md) for the full recipe list.

## 1. Host prerequisites — one-time, needs `sudo`

```bash
sudo apt update
sudo apt install -y podman ffmpeg nvidia-container-toolkit
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
nvidia-ctk cdi list                # expect: nvidia.com/gpu={0,1,all}
```

Verify both GPUs are visible to podman:

```bash
podman run --rm --device nvidia.com/gpu=all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi -L
# → GPU 0: NVIDIA GeForce RTX 3090 ...
# → GPU 1: NVIDIA GeForce RTX 3090 ...
```

## 2. Clone and install the wrappers

```bash
git clone https://github.com/postfix/video-upscale-toolkit.git
cd video-upscale-toolkit
./install.sh                       # copies bin/* → ~/.local/bin/
```

Make sure `~/.local/bin` is on your `$PATH` (the installer warns if not).

## 3. Video2X stack — fast NCNN/Vulkan upscaling

```bash
v2x-install-models                 # ~270 MB community models from GitHub
```

This downloads 6 community Real-ESRGAN models (NMKD-Siax, Remacri, UltraSharp,
HFA2K, etc.) and builds per-task profile dirs.

First upscale (live action, 4× via NMKD-Siax):

```bash
v2x-home some-clip.mp4 some-clip-up.mp4
```

Use both 3090s on one file:

```bash
v2x-nsfw big.mkv big-up.mkv --dual-gpu
```

## 4. Upscale-A-Video stack — diffusion quality for hero clips

Build the image (one-time, ~10 minutes):

```bash
podman build -t localhost/uav:latest containers/uav/
```

Fetch the model weights — Google Drive often rate-limits the anonymous
download, so the easiest path is manual:

```bash
# Open in browser, click Download, accept the "can't scan large file" warning:
xdg-open https://drive.google.com/file/d/17-ZqLJ0gNJGqlO0Mu0Hyoi31fLp0dKWY/view
mv ~/Downloads/upscale_a_video.zip ~/.local/share/uav/models/
uav-install-models                 # unpacks + verifies the model tree
```

First diffusion upscale:

```bash
uav clip.mp4 ./out                 # multi-GPU, bf16, tile_size=128 by default
```

Plan for **~5 seconds of 480 p source → 8–15 minutes** on dual 3090s. UAV is
for short hero clips, not full videos. See [`docs/UAV-NOTES.md`](docs/UAV-NOTES.md)
for the engineering story behind the patches.

## 5. Picking the right tool

| Goal | Tool | Time per minute of source |
|---|---|---|
| Live-action 4× upscale | `v2x-home --dual-gpu` | ~1–3 min |
| Anime episode 4× | `v2x-anime --dual-gpu` | ~1–3 min |
| Old VHS-era footage | `v2x-oldnsfw --deep --dual-gpu` | ~5–15 min |
| Adult content, skin tones | `v2x-nsfw --dual-gpu` | ~1–3 min |
| Hero clip (≤ 60 s), max quality | `uav` | ~1.5–3 h |

If you want a different model on the Video2X side, see the `--model` flag:

```bash
video2x in.mp4 out.mp4 --model ultrasharp     # very sharp, line art / hard edges
video2x in.mp4 out.mp4 --model high-fidelity  # preserves detail in clean sources
video2x in.mp4 out.mp4 --model lsdir-plus     # modern general default
```

## Troubleshooting

- **`v2x-...: profile 'X' not found`** — run `v2x-install-models` first.
- **`uav: models not installed`** — see step 4 above.
- **`vkEnumeratePhysicalDevices failed -3`** — CDI spec missing or stale;
  re-run `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`.
- **UAV produces all-black frames** — the bf16 VAE patch didn't take. Rebuild
  the image: `podman build --no-cache -t localhost/uav:latest containers/uav/`.
- **UAV OOMs at decode** — check `nvidia-smi` for other processes hogging
  GPU 1; UAV's multi-GPU split needs both cards mostly-free.
