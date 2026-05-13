# Nabhya NDVI Backend

FastAPI inference server for the Nabhya NDVI crop health analyzer.

## Local Dev

```bash
# Create venv
python3.11 -m venv .venv && source .venv/bin/activate

# Install deps
pip install -r requirements.txt

# Run locally (model path auto-resolves to ../best_model (1).pth)
uvicorn api_server:app --reload --port 8000
```

Open **http://localhost:8000/docs** for the Swagger UI.

## Environment Variables

| Variable     | Default                           | Purpose                       |
|--------------|-----------------------------------|-------------------------------|
| `MODEL_PATH` | `../best_model (1).pth` (relative)| Absolute path to `.pth` file  |
| `PORT`       | 8000                              | Set automatically by Railway  |

## Endpoints

| Method | Path       | Description                                   |
|--------|------------|-----------------------------------------------|
| GET    | `/`        | Health check + model status                   |
| GET    | `/health`  | Simple liveness probe                         |
| POST   | `/analyze` | Upload image → heatmap + overlay + statistics |

### `/analyze` Response

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
  "heatmap_base64":  "<base64 PNG>",
  "overlay_base64":  "<base64 PNG>",
  "original_base64": "<base64 PNG>"
}
```

## Railway Deployment

1. Push this `backend/` folder to a GitHub repo (e.g. `nabhya-backend`)
2. Railway → New Project → Deploy from GitHub → select repo
3. Set `MODEL_PATH` env var to the path where your `.pth` is stored  
   (upload model via Railway Volume or use HuggingFace Hub to pull on startup)
4. Railway auto-deploys on every push

## Architecture

```
Antigravity / Lovable (frontend)
        ↓  POST /analyze (multipart image)
Railway FastAPI (this server)
        ↓  torch.load()
best_model.pth  (U-Net Generator)
        ↓  returns base64 PNGs + stats
```
