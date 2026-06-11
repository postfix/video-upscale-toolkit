"""Two-GPU placement for UAV — required to fit in 24 GB consumer cards.

V2 layout (sharded UNet — Jun 2026):
    cuda:0  — UNet first half: conv_in + time/class_emb + down_blocks
              + down_temp_blocks                         (~5 GB residual + activations)
              + VAE                                       (~5 GB)
              + text_encoder                              (~1 GB — keep with down_blocks
                so their cross-attention's k/v projections see encoder_hidden_states
                on the same device; the cuda:1 pre-hooks ferry it to mid/up blocks)
    cuda:1  — UNet second half: mid_block + mid_temp_block
              + up_blocks + up_temp_blocks
              + conv_norm_out + conv_act + conv_out      (~7 GB residual + activations)

Why split this way: UAV's UNet hits ~21 GiB peak when fully on cuda:0 at
480p untiled, which forces a 5x4 tile loop on a 24 GB card. With the UNet
sharded, each card's peak halves and untiled 480p actually fits.

VAE moves to cuda:0 (was cuda:1 in v1) because cuda:1 now holds the heavier
half of the UNet. VAE peak is ~5 GiB at tile_size <= 256 input.

Cross-GPU transfers happen at the down/up boundary via forward pre-hooks
registered on every cuda:1 sub-module. The hook is idempotent — Tensor.to(d)
when the tensor is already on d is free. Output of conv_out moves back to
cuda:0 via a post-hook so the rest of the pipeline doesn't care.

Used by the patched inference_upscale_a_video.py via:
    from multi_gpu_setup import setup_multi_gpu
    pipeline = setup_multi_gpu(pipeline)
"""
import copy
import torch


def _to_device_recursive(obj, device):
    """Move Tensors / tuples / lists of Tensors to ``device``. No-op otherwise."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, (tuple, list)):
        moved = type(obj)(_to_device_recursive(x, device) for x in obj)
        return moved
    return obj


def _make_pre_hook(device):
    def hook(module, args, kwargs):
        new_args = tuple(_to_device_recursive(a, device) for a in args)
        new_kwargs = {k: _to_device_recursive(v, device) for k, v in kwargs.items()}
        return new_args, new_kwargs
    return hook


def _make_post_hook(device):
    def hook(module, args, kwargs, output):
        if hasattr(output, "sample"):
            output.sample = _to_device_recursive(output.sample, device)
            return output
        return _to_device_recursive(output, device)
    return hook


def _mem(tag):
    parts = [tag]
    for i in range(torch.cuda.device_count()):
        a = torch.cuda.memory_allocated(i) / 2**30
        parts.append(f"cuda:{i}={a:.2f}GiB")
    print("[mem] " + " ".join(parts))


def setup_multi_gpu(pipeline, dev0="cuda:0", dev1="cuda:1"):
    if torch.cuda.device_count() < 2:
        print(f"[multi_gpu_setup] only {torch.cuda.device_count()} GPU(s) visible — falling back to single-GPU")
        return pipeline.to(dev0)

    unet = pipeline.unet
    d0 = torch.device(dev0)
    d1 = torch.device(dev1)
    _mem("before any moves")

    # ---- shard the UNet ----
    # cuda:0 side — encoder half
    unet.conv_in.to(d0)
    if hasattr(unet, "time_proj"):
        unet.time_proj.to(d0)
    if hasattr(unet, "time_embedding"):
        unet.time_embedding.to(d0)
    if getattr(unet, "class_embedding", None) is not None:
        unet.class_embedding.to(d0)
    # Deepcopy before move so any module instance shared between down/up
    # (UAV's temporal attention modules share a single rotary_emb across stages)
    # becomes independent — otherwise the last .to() wins and the other half
    # gets a cross-device tensor mismatch at first forward.
    unet.down_blocks = copy.deepcopy(unet.down_blocks)
    unet.down_blocks.to(d0)
    _mem("after deepcopy + down_blocks → d0")
    if hasattr(unet, "down_temp_blocks"):
        unet.down_temp_blocks = copy.deepcopy(unet.down_temp_blocks)
        unet.down_temp_blocks.to(d0)
        _mem("after deepcopy + down_temp_blocks → d0")

    # cuda:1 side — decoder half
    unet.mid_block.to(d1)
    _mem("after mid_block → d1")
    if hasattr(unet, "mid_temp_block"):
        unet.mid_temp_block.to(d1)
    unet.up_blocks.to(d1)
    _mem("after up_blocks → d1")
    if hasattr(unet, "up_temp_blocks"):
        unet.up_temp_blocks.to(d1)
        _mem("after up_temp_blocks → d1")
    if hasattr(unet, "conv_norm_out"):
        unet.conv_norm_out.to(d1)
    if hasattr(unet, "conv_act"):
        unet.conv_act.to(d1)
    unet.conv_out.to(d1)

    # ---- register cross-GPU transfer hooks ----
    # Pre-hooks on every cuda:1 sub-module that takes inputs from cuda:0.
    # The hook moves all tensor args/kwargs (including tuples of tensors —
    # res_hidden_states_tuple — and the timesteps/encoder_hidden_states that
    # the pipeline keeps on cuda:0). Idempotent: tensor.to(d1) is free when
    # the tensor is already on d1.
    cuda1_modules = [unet.mid_block]
    if hasattr(unet, "mid_temp_block"):
        cuda1_modules.append(unet.mid_temp_block)
    for blk in unet.up_blocks:
        cuda1_modules.append(blk)
    if hasattr(unet, "up_temp_blocks"):
        for blk in unet.up_temp_blocks:
            cuda1_modules.append(blk)
    for mod_name in ("conv_norm_out", "conv_act", "conv_out"):
        if hasattr(unet, mod_name):
            cuda1_modules.append(getattr(unet, mod_name))

    for m in cuda1_modules:
        m.register_forward_pre_hook(_make_pre_hook(d1), with_kwargs=True)

    # Post-hook on the UNet itself moves the output back to cuda:0 so the
    # pipeline's scheduler / latent math stays on cuda:0 like the original.
    unet.register_forward_hook(_make_post_hook(d0), with_kwargs=True)

    # ---- place the rest of the pipeline ----
    # text_encoder STAYS on cuda:0 — the down_blocks live there and their
    # cross-attention k/v projections need encoder_hidden_states on the same
    # device as the Linear weights. The cuda:1 pre-hooks above will ferry
    # encoder_hidden_states to cuda:1 for the mid/up blocks once per step.
    pipeline.text_encoder.to(d0)
    _mem("after text_encoder → d0")
    if getattr(pipeline, "propagator", None) is not None:
        pipeline.propagator.to(d0)
    pipeline.vae.to(d0)
    _mem("after vae → d0")

    # sanity dump: where does each UNet sub-module actually live?
    devmap = {}
    for name, mod in unet.named_modules():
        devs = {str(p.device) for p in mod.parameters(recurse=False)}
        if devs:
            devmap.setdefault(next(iter(devs)) if len(devs) == 1 else "MIXED", []).append(name or "<root>")
    for dev, names in devmap.items():
        print(f"[devmap] {dev}: {len(names)} submodules, first 3: {names[:3]}")

    # NOTE: do NOT force `_use_memory_efficient_attention_xformers = True`
    # on every attention module — UAV's TemporalAttention has an untested
    # xformers path that produces shape mismatches at runtime (e.g.
    # "mat1 and mat2 shapes cannot be multiplied (18432x4096 and 512x512)").
    # The pipeline.enable_xformers_memory_efficient_attention() called by
    # inference_upscale_a_video.py:134 (Dockerfile patch #6) goes through
    # UAV's set_use_* method, which correctly sets the flag only on
    # attn_spatial — leaving temporal attention on the working standard
    # path. That's the right behaviour; don't second-guess it here.

    print(f"[multi_gpu_setup] V2 sharded UNet: encoder→{dev0}, decoder→{dev1}, VAE+text_encoder→{dev0}")

    # ---- patched VAE decode (now on cuda:0) ----
    vae_dev = d0

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
