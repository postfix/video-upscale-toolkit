# Troubleshooting

Runtime errors we've actually hit on this stack, with root causes and fixes.
The `uav` and `video2x` wrappers auto-detect the common cases — what's here
is for when they don't, or when you want to understand what they're doing.

## UAV: `av.error.BlockingIOError: Resource temporarily unavailable` at video read

**Symptom**

```
[multi_gpu_setup] UNet/text_encoder → cuda:0, VAE → cuda:1
Failed initializing scaling graph (Resource temporarily unavailable):
  fmt:yuv420p csp:... prim:... trc:... -> fmt:rgb24 csp:gbr ...
av.error.BlockingIOError: [Errno 11] Resource temporarily unavailable
  ...torchvision/io/video.py", line 324, in read_video
```

Note that this happens *after* the multi-GPU banner — UAV initialized fine,
the failure is upstream of the model in the video-reading step. Persists even
when the source has perfectly fine `color_primaries / color_trc / colorspace`
metadata — the swscale init genuinely refuses for other reasons (likely a
PyAV 17 / FFmpeg 7 incompatibility specific to our build).

**Root cause**

UAV's `utils.py` calls `torchvision.io.read_video()` → PyAV → swscale. The
upstream UAV requirements pin `av==9.1.0` but that version doesn't build
on Debian 13 (Cython API drift), so our container ships `av` 17.x. The
combination + this image's libav silently fails swscale init for many
real-world inputs. Decord — the obvious alternative — segfaults during
`get_batch()` in this same container image.

**Fix (baked into the container — patch #5 in the Dockerfile)**

The Dockerfile applies `fix_utils.py` at build time, which replaces
`read_frame_from_videos` to use `cv2.VideoCapture` instead. cv2 is the same
reader UAV already uses in the folder-of-images branch of the same function,
so semantics are identical — except cv2 is stable, where torchvision and
decord are not.

After rebuild you should see the patched function:

```bash
podman run --rm --entrypoint head localhost/uav:latest -20 /opt/Upscale-A-Video/utils.py
# expect:  # PATCH: cv2.VideoCapture instead of torchvision.io.read_video.
```

**`uav --retag` / `--no-retag` flags**

The wrapper *also* has an auto-color-tag step (BSF patch for H.264/HEVC,
lossless re-encode for other codecs), cached at `~/.cache/uav/tagged/`. With
the cv2 patch this is mostly cosmetic — cv2 doesn't care about color tags —
but it's cheap (~1 s per file, cached) and harmless. Toggle with:

```bash
uav clip.mp4 ./out --no-retag      # disable auto-tag (smaller cache, same result)
uav clip.mp4 ./out --retag         # force re-tag (debug only)
```

If you ever need to manually pre-tag a video for another tool:

```bash
# H.264 / HEVC — bitstream patch, no re-encode, takes seconds:
ffmpeg -i in.mp4 -map 0 -c copy \
  -bsf:v "h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1" \
  in-tagged.mp4

# Anything else — lossless re-encode:
ffmpeg -i in.mov -map 0 -c:v libx264 -crf 0 -preset ultrafast \
  -color_primaries bt709 -color_trc bt709 -colorspace bt709 \
  -pix_fmt yuv420p -c:a copy in-tagged.mp4
```

---

## UAV: all-black output, no obvious error

**Symptom**

UAV completes, produces a video file of a few KB, but every frame is
uniform YUV black (YMIN = YAVG = YMAX = 16 on `signalstats`). A warning
appears in the log:

```
RuntimeWarning: invalid value encountered in cast
  upscaled_video = upscaled_video.cpu().numpy().astype(np.uint8)
```

**Root cause**

The VAE is in fp16 instead of bf16 — fp16's small exponent range causes the
VAE to overflow during decode, producing NaN values that cast to 0 (YUV
black). The Dockerfile's bf16 patch in `containers/uav/Dockerfile` either
failed to apply (anchor pattern moved upstream) or was reverted.

**Fix**

Rebuild the image; the patch is in the Dockerfile:

```bash
podman build --no-cache -t localhost/uav:latest containers/uav/
```

Verify the patch is in place:

```bash
podman run --rm --entrypoint grep localhost/uav:latest -n \
  "to(dtype=torch.bfloat16)" /opt/Upscale-A-Video/inference_upscale_a_video.py
# expect 2 matches (lines 108 and 113)
```

If `0 matches` → the upstream UAV script changed and the sed anchor needs
updating in the Dockerfile. See `docs/UAV-NOTES.md` for the patch story.

---

## UAV: OOM at `Decoding: 0/N`

**Symptom**

```
Error CUDA out of memory. Tried to allocate XX.XX GiB (GPU 1; 23.56 GiB
  total capacity; YY.YY GiB already allocated; ...
NameError: name 'output_tile' is not defined
```

**Root cause**

Either (a) GPU 1 is being used by something else (Video2X run, X server,
another job) so UAV's VAE doesn't have its full 24 GB budget, or (b) the
patches in `containers/uav/Dockerfile` didn't all apply.

**Diagnose**

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
# Both GPUs should be near 0 / <1 GiB before starting uav.
```

If GPU 1 is busy, free it first or pin UAV to GPU 0:

```bash
uav clip.mp4 ./out --device 0      # but then needs single-card mode — usually OOMs
```

Single-card UAV does not fit in 24 GB. Use dual-3090 mode (the default).

---

## UAV: OOM trying to allocate tens of GiB *before* the first decode

**Symptom**

```
[multi_gpu_setup] UNet/text_encoder → cuda:0, VAE → cuda:1
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 63.45 GiB
  (GPU 0; 23.56 GiB total capacity; 13.10 GiB already allocated; ...
```

Allocation in the tens of GiB, on `cuda:0`, with no `Decoding: N/M` line
ever printed. The bigger the input clip, the bigger the failed allocation.

**Root cause**

`inference_upscale_a_video.py` (~line 217) pre-allocates the *entire*
upscaled output as one float32 tensor on `cuda:0` before the decode loop
starts:

```python
upscaled_video = vframes.new_zeros(output_shape)   # shape (1, 3, T, 4H, 4W)
```

For 4× upscale that's `T × 3 × 4H × 4W × 4 bytes` = `192 × W × H × T`.
A 640×480 @ 30 fps clip works out to ~1.69 GiB **per second** of source —
so a 38 s clip needs 64 GiB on a single card and there's no way to fit it.
This is *separate* from the tile-decode OOM above: it fires before the
VAE ever runs, and tiling/`short_seq` don't help (they affect the decoder
buffer, not the output buffer).

**Fix (default in the wrapper)**

The `uav` wrapper splits the input into chunks small enough that each
chunk's output tensor fits a 10 GiB budget, runs UAV per chunk, and concats
the results losslessly. This is the default — `--chunk auto` — so most
users never see this OOM in the first place.

```bash
uav clip.mp4 ./out                # auto-chunk (default), 10 GiB output budget
uav clip.mp4 ./out --chunk 8      # force 8-second chunks
uav clip.mp4 ./out --no-chunk     # legacy single-shot; OOMs for clips > ~6 s @ 480p

UAV_OUTPUT_BUDGET_BYTES=$((6*1024**3)) uav clip.mp4 ./out   # tighter 6 GiB budget
```

Staging dir lives at `~/.cache/uav/chunks/<basename>.<size-mtime>.c<sec>s/`
and is *kept across runs* so a Ctrl-C or mid-chunk failure resumes from the
last completed chunk on the next invocation. Pass `--clean-chunks` to delete
it after a successful concat.

The math behind the auto-chunk picker is in `docs/UAV-NOTES.md`
("Chunking" section).

---

## Video2X: `--realesrgan-model X is invalid`

**Symptom**

```
[critical] Error parsing arguments: the argument for option 'realesrgan-model' is invalid
```

**Root cause**

Video2X's CLI hardcodes the accepted model names
(`realesrgan-plus | realesrgan-plus-anime | realesr-animevideov3`) and
rejects anything else before checking the filesystem. You passed
`--realesrgan-model remacri` (or another community model name) directly.

**Fix**

Use `--profile NAME` or `--model NAME` instead — these mount a profile dir
where the community model is staged into the standard slot name:

```bash
video2x in.mp4 out.mp4 --model remacri              # auto-resolves to --profile remacri
v2x-nsfw in.mp4 out.mp4                             # preset, already uses the right profile
```

If `--model NAME` fails with "profile not found", run:

```bash
v2x-install-models    # builds all profiles, including ultrasharp / high-fidelity / lsdir-plus / hfa2k / remacri / nmkd-siax
```

---

## Video2X: `vkEnumeratePhysicalDevices failed -3`

**Symptom**

Container loads but reports no Vulkan devices, or errors out at startup
with `failed -3`.

**Root cause**

CDI spec is stale or missing. Happens after NVIDIA driver upgrades.

**Fix**

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
nvidia-ctk cdi list           # expect nvidia.com/gpu={0,1,all}
```

Upstream also has a `--privileged` workaround if CDI still fails:

```bash
video2x in.mp4 out.mp4 --shell    # exec into the container
podman run --rm --gpus all --privileged ghcr.io/k4yt3x/video2x:6.4.0 --list-devices
```

---

## Video2X dual-gpu: lopsided chunk sizes

**Symptom**

`--dual-gpu` splits the input, but one chunk takes much longer than the
other. Wall-clock benefit is small.

**Root cause**

The split happens at I-frame boundaries via `ffmpeg -f segment`. If the
source has rare keyframes (e.g. `keyint=600` at 24 fps = one I-frame per
25 s), the split can land far from the duration midpoint.

**Fix**

Re-encode the source with a tighter GOP before processing:

```bash
ffmpeg -i source.mp4 -c:v libx264 -g 60 -keyint_min 30 -crf 18 \
  -c:a copy source-gop60.mp4
v2x-nsfw source-gop60.mp4 out.mp4 --dual-gpu
```

Or accept the imbalance — it's still faster than single-GPU.

---

## `uav-install-models`: Google Drive rate-limit

**Symptom**

```
Failed to retrieve file url:
  Too many users have viewed or downloaded this file recently. ...
```

**Root cause**

UAV's pretrained weights live in a public Google Drive folder that's heavily
rate-limited at the anonymous-download tier.

**Fix**

Manual download via browser (signed into Google bypasses anon quota):

```
https://drive.google.com/file/d/17-ZqLJ0gNJGqlO0Mu0Hyoi31fLp0dKWY/view
```

Save as `~/.local/share/uav/models/upscale_a_video.zip`, then re-run
`uav-install-models` — it'll detect the local zip and unpack it.
