# Prompts for `uav` (Upscale-A-Video)

`uav` accepts `--prompt` (positive) and `--neg-prompt` (negative). Defaults are
generic ("best quality, extremely detailed" / "blur, worst quality") and rarely
optimal for any specific source. These recipes are tuned per content type.

Diffusion-prompt heuristics that work for video restoration:

- **Describe the source** — give the model context ("1990s home video",
  "low-light scene") so it doesn't reach for stock-photo defaults
- **Describe desired qualities** — texture, grain, lighting, naturalness
- **Negative prompts suppress** the diffusion's tendency to plasticize skin
  and over-sharpen edges; keep them short — too many terms dilute the guidance

## Recipes

### Old VHS / camcorder footage (pre-2005)

```bash
uav vhs.mp4 ./out \
  --prompt "vintage 1990s home video, natural skin tones, soft film grain, warm tungsten lighting, sharp facial features, realistic texture" \
  --neg-prompt "plastic skin, oversharpened, AI hallucination, banding, smearing, distorted faces, oversaturated"
```

### Modern home video — phone / mirrorless / camcorder

```bash
uav clip.mp4 ./out \
  --prompt "natural daylight home video, sharp focus, accurate colors, realistic skin texture, slight film grain" \
  --neg-prompt "plastic skin, oversharpened edges, color bleeding, motion blur, waxy texture"
```

### NSFW / skin-focused content (intimate lighting)

```bash
uav clip.mp4 ./out \
  --prompt "natural skin with visible pores, soft warm lighting, intimate atmosphere, realistic body proportions, film-like quality, fine detail" \
  --neg-prompt "plastic skin, waxy texture, oversmoothed, deformed anatomy, distorted limbs, banding in shadows, AI artifacts"
```

### Anime / cartoon

```bash
uav ep01.mp4 ./out \
  --prompt "high quality anime, clean line art, vibrant saturated colors, detailed eyes and hair, smooth gradients" \
  --neg-prompt "blurry, color bleeding, jagged edges, watermark, deformed features, low quality scan"
```

### Black-and-white footage

```bash
uav bw.mp4 ./out \
  --prompt "vintage black and white film, sharp contrast, fine film grain, natural shadow gradation, period-correct detail" \
  --neg-prompt "color tint, oversharpened, posterization, plastic skin, modern look"
```

### Low-light / night scenes

```bash
uav night.mp4 ./out \
  --prompt "low-light scene with preserved shadow detail, natural color in highlights, soft film grain, no banding" \
  --neg-prompt "noise amplification, banding in dark areas, crushed blacks, color cast, oversaturated highlights"
```

### Generic — content type unknown

```bash
uav unknown.mp4 ./out \
  --prompt "high quality video, sharp focus, accurate colors, natural texture, fine detail preservation" \
  --neg-prompt "blur, compression artifacts, banding, plastic, oversharpened, AI hallucination"
```

## Faces and skin: UAV is conservative by default

UAV restores backgrounds, textures, fabric, and architectural detail
aggressively. **It is deliberately conservative on human faces, bodies, and
skin** — most diffusion video upscalers are. The model treats faces as a
strong prior to avoid identity drift and uncanny-valley artifacts. With
the default `-n 120 -g 6 --prompt "best quality, extremely detailed"`,
expect background detail to jump a tier while faces look essentially
unchanged from the source.

**To push UAV harder on skin/face regions**, all three dials matter:

```bash
uav clip.mp4 ./out -n 160 -g 8 -s 30 \
  --prompt "high resolution skin texture, sharp facial features, visible pores, detailed eyes, natural skin tones, fine hair detail" \
  --neg-prompt "plastic skin, waxy texture, smoothed face, blurry features, AI hallucination, distorted anatomy"
```

- `-n 160` (vs default 120) tells the scheduler to alter the source more
- `-g 8` (vs default 6) makes the prompt drive harder over the source prior
- Explicit face/skin tokens in the prompt are the only way the model knows
  to recover those regions specifically

Even tuned this way UAV will not match a face-specialised model. **For
serious face restoration on this hardware, use a two-pass workflow:**

1. UAV pass for overall texture/background (preserves identity)
2. **GFPGAN or CodeFormer** as a second pass to enhance face regions

GFPGAN/CodeFormer are feed-forward (CNN), not diffusion — they finish a
38 s clip in ~30 s on a 3090, vs UAV's many hours. They're purpose-built
for face super-resolution and don't have UAV's identity-preserving bias.
Not yet wrapped in this toolkit — track issue for a `v2x-face` preset.

## Tips

- **Be specific about era.** "1980s home video" produces noticeably more
  period-correct skin tones than the default prompt.
- **Avoid "ultra HD / 8K"** in positive prompts — pushes the diffusion toward
  modern stock-photo look that clashes with old source material.
- **Keep negative short.** Five to seven terms is usually enough; longer
  lists dilute each individual term's effect on guidance.
- **For short hero clips you care about, run twice** with `-n 100` (gentler
  denoise) vs `-n 150` (stronger). `-n` is the single strongest dial UAV
  exposes; the prompt matters less if `-n` is wrong for the source.
- **The `-g` (guidance scale) dial.** Default 6 is a good middle. Lower
  (3-4) lets the prompt influence things less — more faithful to source.
  Higher (8-10) lets the prompt drive harder — useful when the source is
  so degraded the model needs strong guidance toward "what this should be".

## Tested combinations on this hardware (dual 3090, 24 GB ea.)

| Source | Prompt recipe | `-n` | `-g` | `-s` | Notes |
|---|---|---|---|---|---|
| 480p VHS rip, well-lit interior | "Old VHS" | 120 | 6 | 30 | Default — solid starting point |
| 720p phone video, daylight | "Modern home video" | 80 | 5 | 25 | Lower `-n` keeps original texture |
| 720p phone, indoor low light | "Low-light" | 140 | 7 | 35 | Higher denoise needed |
| 480p web rip, NSFW | "NSFW skin" | 120 | 6 | 30 | Watch for plasticized skin → drop `-g` to 4 |
| 720p anime stream rip | "Anime" | 120 | 6 | 30 | `v2x-anime-hd` may be a better choice — UAV can soften linework |
