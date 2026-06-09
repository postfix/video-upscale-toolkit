# Upscale-A-Video container (https://github.com/sczhou/Upscale-A-Video)
# Diffusion-based video super-resolution, PyTorch 2.0.1 + CUDA 11.7.
# Models are NOT baked into the image — bind-mount them at runtime to
# /opt/Upscale-A-Video/pretrained_models/upscale_a_video (use the
# uav-install-models script to fetch ~10 GB from Google Drive once).
FROM docker.io/pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

LABEL org.opencontainers.image.source="https://github.com/sczhou/Upscale-A-Video"
LABEL org.opencontainers.image.description="Upscale-A-Video diffusion-based video super-resolution"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/cache/hf
ENV TRANSFORMERS_CACHE=/cache/hf
ENV XDG_CACHE_HOME=/cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libgl1 libglib2.0-0 ca-certificates unzip \
        pkg-config \
        libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
        libswscale-dev libswresample-dev libavfilter-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone --depth=1 https://github.com/sczhou/Upscale-A-Video.git

WORKDIR /opt/Upscale-A-Video

# requirements.txt pins ancient versions; loosen the unbuildable ones:
#   - torch       : already in base image
#   - av==9.1.0   : no py3.10 wheel + 2022-era Cython; unpin to get a recent wheel
#   - xformers    : pin to ==0.0.20 so we don't drift onto a torch-2.1-only build
RUN sed -i \
        -e '/^torch==/d' \
        -e 's/^av==.*/av/' \
        -e 's/^xformers.*/xformers==0.0.20/' \
        requirements.txt && \
    pip install --no-cache-dir -r requirements.txt && \
    # diffusers 0.16 calls cached_download() which was removed from
    # huggingface_hub >= 0.20; pin to the matching vintage. Same for
    # transformers 4.28 which expects the legacy hub API.
    pip install --no-cache-dir 'huggingface_hub<0.20' && \
    pip install --no-cache-dir gdown==5.2.0

# Memory patches for 2× 24 GB GPUs (e.g. dual RTX 3090):
#
# 1. Two-GPU sharding: UNet+text_encoder on cuda:0, VAE on cuda:1. Each card
#    has the full 24 GB budget for its part of the pipeline. Implemented in
#    multi_gpu_setup.py (copied in below) and wired into inference script.
#
# 2. Skip the FP32 VAE cast — VAE decode peaks at ~15 GB in fp16; with the
#    cast it doubles to ~30 GB and won't fit on a single card. Quality risk
#    on certain content; if you see banding/black frames, restore the cast
#    by removing the second sed below and rebuild.
COPY multi_gpu_setup.py /opt/Upscale-A-Video/multi_gpu_setup.py
RUN sed -i \
    's|^    pipeline = pipeline\.to(UAV_device)$|    from multi_gpu_setup import setup_multi_gpu; pipeline = setup_multi_gpu(pipeline)|' \
    /opt/Upscale-A-Video/inference_upscale_a_video.py && \
    grep -q "setup_multi_gpu" /opt/Upscale-A-Video/inference_upscale_a_video.py && \
    # The UAV inference script replaces pipeline.vae with a fresh AutoencoderKLVideo
    # but never calls .half() on it (the UNet does at line 117). Result: VAE stays
    # in FP32 even though the pipeline was loaded with torch_dtype=fp16. Fix it by
    # injecting .half() after each VAE load_state_dict.
    sed -i \
    's|^        pipeline\.vae\.load_state_dict(.*)$|&\n        pipeline.vae = pipeline.vae.to(dtype=torch.bfloat16)  # PATCH: bf16 — same memory as fp16 but fp32-equivalent range|' \
    /opt/Upscale-A-Video/inference_upscale_a_video.py && \
    [ "$(grep -c 'pipeline.vae = pipeline.vae.to(dtype=torch.bfloat16)' /opt/Upscale-A-Video/inference_upscale_a_video.py)" = "2" ] && \
    sed -i \
    's|^        self\.vae\.to(dtype=torch\.float32)$|        # PATCH: keep VAE in fp16 to fit 24 GB GPUs — was self.vae.to(dtype=torch.float32)|' \
    /opt/Upscale-A-Video/models_video/pipeline_upscale_a_video.py && \
    grep -q "PATCH: keep VAE in fp16" /opt/Upscale-A-Video/models_video/pipeline_upscale_a_video.py && \
    # Reduce VAE decode temporal chunk from 3 frames to 1 frame. The temporal
    # cross-attention buffer scales quadratically with frames-per-chunk (T²),
    # so 3 → 1 cuts that buffer by 9× — typically ~12 GB → ~1.3 GB at
    # tile_size=128. Costs ~3× more decode-loop iterations but each is tiny.
    sed -i \
    's|^        short_seq = 3$|        short_seq = 1  # PATCH: shrink VAE temporal attn buffer for 24 GB GPUs|' \
    /opt/Upscale-A-Video/models_video/pipeline_upscale_a_video.py && \
    grep -q "PATCH: shrink VAE temporal" /opt/Upscale-A-Video/models_video/pipeline_upscale_a_video.py

# stage the model mount point so the bind-mount has a writable home
RUN mkdir -p /opt/Upscale-A-Video/pretrained_models \
             /workspace/input /workspace/output /cache/hf

VOLUME ["/opt/Upscale-A-Video/pretrained_models", "/cache"]

ENTRYPOINT ["python", "inference_upscale_a_video.py"]
