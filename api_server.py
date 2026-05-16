"""
Nabhya NDVI Analysis API
FastAPI backend that loads a trained PyTorch Generator model
and returns NDVI heatmap overlays for uploaded satellite images.

Model source priority:
  1. HuggingFace Hub  — set HF_REPO_ID + HF_TOKEN (+ optionally HF_FILENAME)
  2. Local path       — set MODEL_PATH env var (or uses sibling best_model.pth)
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
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


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
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

    orig_resized = orig.resize((INPUT_SIZE, INPUT_SIZE))
    OUTPUT_SIZE = 512  # upscale for sharper display

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

    return JSONResponse({
        "status": "success",
        "model_used": model is not None,
        "statistics": stats,
        "heatmap_base64":  encode_image(heatmap_img),
        "overlay_base64":  encode_image(overlay),
        "original_base64": encode_image(orig_display),
    })
