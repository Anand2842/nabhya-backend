---
title: Nabhya NDVI API
emoji: 🛰️
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Nabhya NDVI API

FastAPI inference server — satellite image → full-resolution NDVI heatmap.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Status + model info |
| GET | `/health` | Liveness probe |
| POST | `/analyze` | Upload image → heatmap + overlay + stats |

## `/analyze` Response

```json
{
  "status": "success",
  "model_used": true,
  "statistics": {
    "mean_ndvi": 0.42,
    "max_ndvi": 0.91,
    "min_ndvi": 0.03,
    "healthy_pct": 61.5,
    "stressed_pct": 24.3,
    "barren_pct": 14.2
  },
  "heatmap_base64":  "<base64 PNG — full 256×256>",
  "overlay_base64":  "<base64 PNG — full 256×256>",
  "original_base64": "<base64 PNG — full 256×256>"
}
```

## Space Secrets Required

Set in Space → Settings → Variables and secrets:

| Secret | Value |
|--------|-------|
| `HF_TOKEN` | your HuggingFace read token |

Model is pulled from `Anand2842/ndvi_ieee` at startup.
