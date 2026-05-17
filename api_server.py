"""
Nabhya NDVI Analysis API
FastAPI backend that loads a trained PyTorch Generator model
and returns NDVI heatmap overlays for uploaded satellite images.

Model source priority:
  1. HuggingFace Hub  — set HF_REPO_ID + HF_TOKEN (+ optionally HF_FILENAME)
  2. Local path       — set MODEL_PATH env var (or uses sibling best_model.pth)
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import io
import base64
import logging
import os
import time
import cv2
from scipy import ndimage

try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Nabhya NDVI API",
    description="Satellite image → NDVI heatmap using a trained GAN Generator",
    version="1.0.0"
)

# ── CORS (allow Antigravity / Lovable frontend) ──────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Keys ─────────────────────────────────────────────────────────────────
VALID_KEYS = {
    "nbhya_demo_key_001": {"plan": "starter", "limit": 2000, "used": 0},
    "nbhya_test_key_002": {"plan": "growth",  "limit": 6000, "used": 0},
}

def verify_key(request: Request):
    key = request.headers.get("X-Nabhya-Key")
    if not key:
        return None, "API key required"
    if key not in VALID_KEYS:
        return None, "Invalid API key"
    account = VALID_KEYS[key]
    if account["used"] >= account["limit"]:
        return None, "Monthly limit reached"
    return account, None

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Running on device: {DEVICE}")

# ── Model resolution ─────────────────────────────────────────────────────────
# Env vars (set these as Railway secrets — never hardcode):
#   HF_REPO_ID   e.g.  Anand2842/nabhya-ndvi
#   HF_FILENAME  e.g.  best_model.pth          (default: best_model.pth)
#   HF_TOKEN     your HuggingFace read token
#   MODEL_PATH   local fallback path

def resolve_model_path() -> str:
    """Download from HF Hub if env vars present, else fall back to local path."""
    repo_id  = os.environ.get("HF_REPO_ID",   "Anand2842/ndvi_ieee").strip()
    hf_token = os.environ.get("HF_TOKEN",      "").strip()
    filename = os.environ.get("HF_FILENAME",   "best_model (1).pth").strip()

    if repo_id and HF_HUB_AVAILABLE:
        logger.info(f"Downloading model from HuggingFace Hub: {repo_id}/{filename}")
        local = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=hf_token or None,
        )
        logger.info(f"Model cached at: {local}")
        return local

    # Local fallback
    local_path = os.environ.get(
        "MODEL_PATH",
        os.path.join(os.path.dirname(__file__), "..", "best_model (1).pth")
    )
    logger.info(f"Using local model path: {local_path}")
    return local_path

# ── Generator Architecture ────────────────────────────────────────────────────
# Exact architecture from the Kaggle evaluation notebook (Nabhya model).
# Layer names (e1..e7, d1..d7) and internal structure (conv/dropout)
# MUST match the checkpoint's state_dict keys exactly.

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, use_dropout=False):
        super().__init__()
        if down:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True))
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(0.5) if use_dropout else nn.Identity()
    def forward(self, x):
        return self.dropout(self.conv(x))


class Generator(nn.Module):
    """
    Pix2Pix-style U-Net Generator: RGB (3ch) → 3-channel NDVI visualization.
    Architecture matches the Kaggle evaluation checkpoint exactly.
    Input:  (B, 3, 256, 256)
    Output: (B, 3, 256, 256)  values in [-1, 1]
    """
    def __init__(self):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2))
        self.e2 = UNetBlock(64, 128)
        self.e3 = UNetBlock(128, 256)
        self.e4 = UNetBlock(256, 512)
        self.e5 = UNetBlock(512, 512)
        self.e6 = UNetBlock(512, 512)
        self.e7 = UNetBlock(512, 512)
        self.bottleneck = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1), nn.ReLU())
        self.d1 = UNetBlock(512, 512, False, True)
        self.d2 = UNetBlock(1024, 512, False, True)
        self.d3 = UNetBlock(1024, 512, False, True)
        self.d4 = UNetBlock(1024, 512, False)
        self.d5 = UNetBlock(1024, 256, False)
        self.d6 = UNetBlock(512, 128, False)
        self.d7 = UNetBlock(256, 64, False)
        self.final = nn.Sequential(nn.ConvTranspose2d(128, 3, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        b = self.bottleneck(e7)
        d1 = self.d1(b)
        d2 = self.d2(torch.cat([d1, e7], 1))
        d3 = self.d3(torch.cat([d2, e6], 1))
        d4 = self.d4(torch.cat([d3, e5], 1))
        d5 = self.d5(torch.cat([d4, e4], 1))
        d6 = self.d6(torch.cat([d5, e3], 1))
        d7 = self.d7(torch.cat([d6, e2], 1))
        return self.final(torch.cat([d7, e1], 1))


# ── Kaggle-style output conversion ────────────────────────────────────────────
def to_ndvi_image(tensor_output: 'torch.Tensor') -> np.ndarray:
    """Convert model output tensor to [0,1] RGB numpy array.
    Matches the Kaggle evaluation notebook's to_numpy function exactly.
    """
    img = tensor_output.squeeze(0).cpu().detach().numpy()   # (3, H, W)
    img = np.transpose(img, (1, 2, 0))                     # (H, W, 3)
    return np.clip(img * 0.5 + 0.5, 0, 1)                  # [-1,1] → [0,1]


def ndvi_from_rgb(pred_rgb: np.ndarray) -> np.ndarray:
    """Derive a single-channel NDVI proxy from the model's 3-channel output.
    Uses luminance weighting to extract vegetation intensity.
    """
    return 0.2989 * pred_rgb[:, :, 0] + 0.5870 * pred_rgb[:, :, 1] + 0.1140 * pred_rgb[:, :, 2]


# ── NDVI colormap (RdYlGn — multi-stop, matplotlib-equivalent) ────────────────
# Kept as fallback for the mock/no-model path
def ndvi_colormap(ndvi_norm: np.ndarray) -> np.ndarray:
    """Map [0,1] NDVI values to a proper RdYlGn RGB heatmap with 6 color stops."""
    ndvi_norm = np.clip(ndvi_norm, 0, 1)
    stops = np.array([
        [0.0,  0.647, 0.059, 0.082],  # deep red
        [0.2,  0.906, 0.259, 0.204],  # red
        [0.4,  0.992, 0.682, 0.318],  # orange
        [0.5,  1.000, 1.000, 0.600],  # yellow
        [0.7,  0.651, 0.851, 0.416],  # yellow-green
        [0.85, 0.263, 0.671, 0.278],  # green
        [1.0,  0.004, 0.408, 0.216],  # dark green
    ])
    r = np.interp(ndvi_norm, stops[:, 0], stops[:, 1])
    g = np.interp(ndvi_norm, stops[:, 0], stops[:, 2])
    b = np.interp(ndvi_norm, stops[:, 0], stops[:, 3])
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def postprocess_ndvi_image(pred_rgb: np.ndarray) -> np.ndarray:
    """Light postprocessing for the 3-channel model output.
    Uses gentle histogram stretch (1st–99th percentile) per channel.
    """
    result = np.copy(pred_rgb)
    for ch in range(3):
        p1 = np.percentile(result[:, :, ch], 1)
        p99 = np.percentile(result[:, :, ch], 99)
        if (p99 - p1) > 0.01:
            result[:, :, ch] = (result[:, :, ch] - p1) / (p99 - p1)
    result = np.clip(result, 0, 1)
    return result


# ── Model loading ─────────────────────────────────────────────────────────────
model: Generator | None = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        model_path = resolve_model_path()
        logger.info(f"Loading checkpoint: {model_path}")
        checkpoint = torch.load(model_path, map_location=DEVICE)

        gen = Generator().to(DEVICE)

        # Checkpoint key: "generator_state" (verified from Kaggle evaluation notebook)
        if isinstance(checkpoint, dict):
            sd = (
                checkpoint.get("generator_state")
                or checkpoint.get("generator_state_dict")
                or checkpoint.get("model_state_dict")
                or checkpoint.get("state_dict")
                or checkpoint.get("gen")
                or checkpoint
            )
        else:
            sd = checkpoint

        # Log checkpoint info if available
        if isinstance(checkpoint, dict):
            epoch = checkpoint.get("epoch", "?")
            g_loss = checkpoint.get("g_loss", "?")
            logger.info(f"Checkpoint info — epoch: {epoch}, g_loss: {g_loss}")
            logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")

        gen.load_state_dict(sd, strict=True)  # strict=True to catch architecture mismatches
        gen.eval()
        model = gen
        logger.info("✅ Model loaded successfully (strict=True, all weights matched)")
    except Exception as e:
        logger.error(f"❌ Model load failed: {e}")
        logger.warning("Running with MOCK predictions — set HF_REPO_ID + HF_TOKEN or MODEL_PATH.")
        model = None


# ── Image pre-processing ──────────────────────────────────────────────────────
INPUT_SIZE = 256
transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def encode_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def compute_statistics(ndvi_single: np.ndarray) -> dict:
    """Compute NDVI statistics from a single-channel [0,1] NDVI proxy."""
    return {
        "mean_ndvi":    round(float(np.mean(ndvi_single)), 4),
        "max_ndvi":     round(float(np.max(ndvi_single)), 4),
        "min_ndvi":     round(float(np.min(ndvi_single)), 4),
        # Thresholds applied to luminance-derived NDVI proxy
        "healthy_pct":  round(float(np.mean(ndvi_single > 0.6)) * 100, 2),
        "stressed_pct": round(float(np.mean((ndvi_single >= 0.3) & (ndvi_single <= 0.6))) * 100, 2),
        "barren_pct":   round(float(np.mean(ndvi_single < 0.3)) * 100, 2),
    }

# ── Vegetation Intelligence & Stress Zones ────────────────────────────────────

def compute_health_score(healthy_pct, stressed_pct):
    return round(healthy_pct * 0.75 + stressed_pct * 0.25, 1)

def compute_health_grade(score):
    if score >= 85: return "A"
    if score >= 70: return "B"
    if score >= 55: return "C"
    if score >= 40: return "D"
    return "F"

def compute_action_flag(grade):
    if grade in ["A", "B"]: return "HEALTHY"
    if grade == "C": return "MONITOR"
    return "ALERT"

def stress_severity_index(ndvi_mean, field_average):
    """
    Returns 0-100 score where:
    0   = no stress
    100 = maximum stress
    field_average is computed from the actual NDVI array, not hardcoded.
    """
    relative_stress = max(0, field_average - ndvi_mean)
    index = min(100, relative_stress * 200)
    return round(index, 1)

def detect_stress_zones(ndvi_array, resolution_m=10):
    """
    ndvi_array: float32 array 0-1, output of your model (near-NDVI proxy).
    resolution_m: metres per pixel (assume 10m for Sentinel scale)
    NOTE: threshold 0.35 was chosen empirically for our near-NDVI proxy.
    Absolute values may not map exactly to true NDVI. Tuneable later.
    """
    # Threshold — stressed pixels (empirical for near-NDVI proxy)
    stressed_mask = (ndvi_array < 0.35).astype(np.uint8)
    
    # Remove noise — small isolated pixels
    kernel = np.ones((2,2), np.uint8)
    stressed_mask = cv2.morphologyEx(stressed_mask, 
                                      cv2.MORPH_OPEN, kernel)
    
    # Label connected components — each cluster = one zone
    labeled, num_zones = ndimage.label(stressed_mask)
    
    zones = []
    field_avg = float(ndvi_array.mean())  # Actual field average, not hardcoded
    for zone_id in range(1, num_zones + 1):
        zone_pixels = (labeled == zone_id)
        pixel_count = zone_pixels.sum()
        
        # Filter tiny zones — ~0.5 hectares minimum at Sentinel 10m resolution
        if pixel_count < 50:
            continue
        
        # Zone stats
        zone_ndvi = ndvi_array[zone_pixels]
        ndvi_mean = float(zone_ndvi.mean())
        ndvi_std  = float(zone_ndvi.std())
        
        # Area in hectares
        area_sqm = pixel_count * (resolution_m ** 2)
        area_ha   = round(area_sqm / 10000, 2)
        
        # Severity
        if ndvi_mean < 0.2:
            severity = "CRITICAL"
        elif ndvi_mean < 0.3:
            severity = "HIGH"
        else:
            severity = "MODERATE"
        
        # Bounding box for frontend to draw rectangle
        rows = np.where(zone_pixels.any(axis=1))[0]
        cols = np.where(zone_pixels.any(axis=0))[0]
        bbox = {
            "x_min": int(cols.min()),
            "y_min": int(rows.min()),
            "x_max": int(cols.max()),
            "y_max": int(rows.max()),
        }
        
        
        zones.append({
            "zone_id": f"Z{zone_id:03d}",
            "severity": severity,
            "stress_severity_index": stress_severity_index(ndvi_mean, field_avg),
            "ndvi_mean": round(ndvi_mean, 3),
            "ndvi_std":  round(ndvi_std, 3),
            "area_ha":   area_ha,
            "bbox":      bbox
        })
    
    # Sort by severity then area
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2}
    zones.sort(key=lambda z: (severity_order[z["severity"]], -z["area_ha"]))
    
    # Cap at top 15 zones to avoid visual noise
    zones = zones[:15]
    
    # Re-index to ensure sequential zone IDs after filtering
    for i, z in enumerate(zones):
        z["zone_id"] = f"Z{i+1:03d}"
        
    return zones

def draw_zones_on_heatmap(heatmap_img, zones, scale_factor=2):
    img = np.array(heatmap_img) # RGB
    
    colors = {
        "CRITICAL": (214, 40, 57),    # red
        "HIGH":     (244, 167, 38),   # amber  
        "MODERATE": (144, 190, 109),  # light green
    }
    
    for zone in zones:
        bb = zone["bbox"]
        color = colors[zone["severity"]]
        
        # Scale bounding box to match the output image size
        x_min = bb["x_min"] * scale_factor
        y_min = bb["y_min"] * scale_factor
        x_max = bb["x_max"] * scale_factor
        y_max = bb["y_max"] * scale_factor
        
        # Draw white outline first (2px wider)
        cv2.rectangle(img, 
                      (x_min - 2, y_min - 2), 
                      (x_max + 2, y_max + 2), 
                      (255, 255, 255), 1)
        
        # Then coloured rectangle on top
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 3)
        
        # Label setup
        label = f"{zone['zone_id']} {zone['severity']}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        
        # Get text size
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Black background box for label
        cv2.rectangle(img,
            (x_min + 3, y_min + 3),
            (x_min + tw + 9, y_min + th + 9),
            (0, 0, 0), -1)  # filled black
        
        # White text on top
        cv2.putText(img, label,
            (x_min + 6, y_min + th + 5),
            font, font_scale,
            (255, 255, 255), thickness)
    
    return Image.fromarray(img)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return {
        "message": "Nabhya NDVI API Running",
        "model_loaded": model is not None,
        "device": str(DEVICE),
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": model is not None}


@app.get("/api-docs")
async def api_docs():
    return FileResponse("docs.html")


@app.post("/analyze")
async def analyze(request: Request, file: UploadFile = File(...)):
    """
    Upload a satellite image (JPG/PNG) and receive:
    - NDVI heatmap (base64 PNG)
    - Overlay image blended with original (base64 PNG)
    - Per-pixel NDVI statistics
    """
    valid_exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    if not file.content_type.startswith("image/") and not file.filename.lower().endswith(valid_exts):
        raise HTTPException(400, "Only image files are accepted (JPEG/PNG/TIFF).")

    contents = await file.read()
    try:
        orig = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Cannot decode uploaded file as an image.")

    account, err_msg = verify_key(request)
    # Note: Opt-in auth logic. If no key, treated as demo mode.
    # We could reject unauthenticated users here by checking: if not account and key_was_provided...
    # For now, allow unauthenticated for demo compatibility.
    if account:
        account["used"] += 1

    orig_resized = orig.resize((INPUT_SIZE, INPUT_SIZE))
    OUTPUT_SIZE = 512  # upscale for sharper display

    t0 = time.time()
    # ── Inference ────────────────────────────────────────────────────────────
    if model is not None:
        tensor = transform(orig_resized).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = model(tensor)                       # (1, 3, H, W) in [-1, 1]

        # ── Kaggle-style denormalization: (x+1)/2, NO inversion ──────────────
        pred_rgb = to_ndvi_image(pred)                 # (H, W, 3) in [0, 1]

        # ── Derive single-channel NDVI for statistics ────────────────────────
        ndvi_single = ndvi_from_rgb(pred_rgb)

        # ── Statistics on raw NDVI (before any postprocessing) ────────────────
        stats = compute_statistics(ndvi_single)

        # ── Preserve raw NDVI for zone detection (before histogram stretch) ──
        ndvi_raw = ndvi_single.copy()

        # ── Apply RdYlGn colormap to single-channel NDVI for farmer-friendly visualization
        p1, p99 = np.percentile(ndvi_single, 1), np.percentile(ndvi_single, 99)
        if (p99 - p1) > 0.01:
            ndvi_single = (ndvi_single - p1) / (p99 - p1)
        ndvi_single = np.clip(ndvi_single, 0, 1)
        heatmap_rgb = ndvi_colormap(ndvi_single)
        heatmap_img = Image.fromarray(heatmap_rgb)
    else:
        # Graceful mock: compute rough NDVI-like map from R/G channels
        arr = np.array(orig_resized).astype(np.float32) / 255.0
        nir_approx = arr[:, :, 1]          # green ≈ NIR proxy
        red = arr[:, :, 0]
        denom = nir_approx + red + 1e-6
        ndvi_arr = np.clip((nir_approx - red) / denom, 0, 1)

        stats = compute_statistics(ndvi_arr)

        # Apply colormap for the mock path
        heatmap_rgb = ndvi_colormap(ndvi_arr)
        heatmap_img = Image.fromarray(heatmap_rgb)

    # ── Upscale 256 → 512 (sharper display without re-running model) ─────────
    heatmap_img = heatmap_img.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS)
    orig_display = orig_resized.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS)

    # ── Light saturation boost ×1.2 (gentler than before) ────────────────────
    from PIL import ImageEnhance
    heatmap_img = ImageEnhance.Color(heatmap_img).enhance(1.2)

    # ── Blended overlay ───────────────────────────────────────────────────────
    overlay = Image.blend(orig_display.convert("RGB"), heatmap_img, alpha=0.55)

    inference_time_ms = int((time.time() - t0) * 1000)

    # ── Compute Intelligence & Zones ──────────────────────────────────────────
    # Use raw (pre-stretch) NDVI for zone detection so thresholds are meaningful
    ndvi_array_for_zones = ndvi_raw if model is not None else ndvi_arr
    zones = detect_stress_zones(ndvi_array_for_zones, resolution_m=10)
    
    # Draw zones on upscaled heatmap
    annotated_heatmap = draw_zones_on_heatmap(heatmap_img, zones, scale_factor=2) # 256→512 upscale

    # Overall metrics
    score = compute_health_score(stats["healthy_pct"], stats["stressed_pct"])
    grade = compute_health_grade(score)
    flag = compute_action_flag(grade)

    if stats["healthy_pct"] > 70:
        dominant_condition = "Healthy with moderate stress patches" if stats["stressed_pct"] > 10 else "Optimal vegetation health"
    elif stats["stressed_pct"] > 40:
        dominant_condition = "Widespread vegetation stress detected"
    else:
        dominant_condition = "Mixed health, significant bare soil areas"

    total_stressed_ha = sum(z["area_ha"] for z in zones)

    return JSONResponse({
        "status": "success",
        "nabhya_version": "1.0",
        "auth": "verified" if account else "demo",
        "inference_time_ms": inference_time_ms,
        "image_resolution": f"{OUTPUT_SIZE}x{OUTPUT_SIZE}",
        "vegetation_intelligence": {
            "overall_health_score": score,
            "health_grade": grade,
            "zones": {
                "healthy_pct": stats["healthy_pct"],
                "stressed_pct": stats["stressed_pct"],
                "barren_pct": stats["barren_pct"]
            },
            "dominant_condition": dominant_condition,
            "action_flag": flag
        },
        "stress_zones": zones,
        "total_stress_zones": len(zones),
        "total_stressed_ha": round(total_stressed_ha, 2),
        "heatmap_base64": encode_image(heatmap_img),
        "annotated_heatmap_base64": encode_image(annotated_heatmap),
        "overlay_base64": encode_image(overlay),
        "original_base64": encode_image(orig_display),
        # Legacy key for backward compat
        "statistics": stats,
        "model_used": model is not None,
        "metadata": {
            "model": "Pix2Pix UNet" if model is not None else "Mock RGB Logic",
            "training_images": 2200 if model is not None else 0,
            "ssim_benchmark": 0.8060 if model is not None else None,
            "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
    })
