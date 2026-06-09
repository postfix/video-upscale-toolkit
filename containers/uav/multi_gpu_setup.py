"""Two-GPU placement for UAV — required to fit in 24 GB consumer cards.

Layout:
    cuda:0  — UNet, text_encoder, propagator    (~18 GB peak)
    cuda:1  — VAE                                (~15 GB peak in fp16)

Cross-GPU transfers happen in the patched `decode_latents` / `decode_latents_vsr`
methods: latents move cuda:0 → cuda:1, the decoded image moves cuda:1 → cuda:0.

Used by the patched inference_upscale_a_video.py via:
    from multi_gpu_setup import setup_multi_gpu
    pipeline = setup_multi_gpu(pipeline)
"""
import torch


def setup_multi_gpu(pipeline, unet_device="cuda:0", vae_device="cuda:1"):
    if torch.cuda.device_count() < 2:
        print(f"[multi_gpu_setup] only {torch.cuda.device_count()} GPU(s) visible — falling back to single-GPU")
        return pipeline.to(unet_device)

    pipeline.unet         = pipeline.unet.to(unet_device)
    pipeline.text_encoder = pipeline.text_encoder.to(unet_device)
    if getattr(pipeline, "propagator", None) is not None:
        pipeline.propagator = pipeline.propagator.to(unet_device)
    pipeline.vae          = pipeline.vae.to(vae_device)

    print(f"[multi_gpu_setup] UNet/text_encoder → {unet_device}, VAE → {vae_device}")

    vae_dev = torch.device(vae_device)

    # IMPORTANT: cast latents to VAE dtype (fp16). The pipeline's __call__
    # converts latents to fp32 around line 669 ("VAE overflows in float16")
    # but our FP32-cast patch (line 668) keeps the VAE in fp16. If we don't
    # match the dtypes here, PyTorch promotes the entire VAE to fp32 at the
    # first conv, blowing the memory budget.
    def patched_decode_latents(latents):
        orig_device = latents.device
        vae_dtype = pipeline.vae.dtype
        latents = latents.to(device=vae_dev, dtype=vae_dtype)
        latents = 1 / pipeline.vae.config.scaling_factor * latents
        image = pipeline.vae.decode(latents).sample
        torch.cuda.empty_cache()
        return image.to(orig_device)

    def patched_decode_latents_vsr(latents, img, w_lr):
        orig_device = latents.device
        vae_dtype = pipeline.vae.dtype
        latents = latents.to(device=vae_dev, dtype=vae_dtype)
        img = img.to(device=vae_dev, dtype=vae_dtype)
        if isinstance(w_lr, torch.Tensor):
            w_lr = w_lr.to(device=vae_dev, dtype=vae_dtype)
        latents_scaled = 1 / pipeline.vae.config.scaling_factor * latents
        image = pipeline.vae.decode(latents_scaled, img, w_lr).sample
        torch.cuda.empty_cache()
        return image.to(orig_device)

    pipeline.decode_latents     = patched_decode_latents
    pipeline.decode_latents_vsr = patched_decode_latents_vsr

    return pipeline
