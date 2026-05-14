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
# Standard U-Net-style generator commonly used in pix2pix / NDVI GAN papers.
# If your architecture differs, swap in your own Generator class here.

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_bn=True, dropout=False, relu=True):
        super().__init__()
        layers = []
        if down:
            layers.append(nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=not use_bn))
        else:
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=not use_bn))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if relu:
            layers.append(nn.ReLU(inplace=True) if not down else nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    """
    Pix2Pix-style U-Net Generator: RGB (3ch) → single-channel NDVI map.
    Input:  (B, 3, 256, 256)
    Output: (B, 1, 256, 256)  values in [-1, 1], mapped to [0, 1] NDVI
    """
    def __init__(self, in_channels=3, out_channels=1, features=64):
        super().__init__()
        # Encoder
        self.down1 = nn.Sequential(nn.Conv2d(in_channels, features, 4, 2, 1), nn.LeakyReLU(0.2))
        self.down2 = UNetBlock(features,     features*2)
        self.down3 = UNetBlock(features*2,   features*4)
        self.down4 = UNetBlock(features*4,   features*8)
        self.down5 = UNetBlock(features*8,   features*8)
        self.down6 = UNetBlock(features*8,   features*8)
        self.down7 = UNetBlock(features*8,   features*8)
        self.bottleneck = nn.Sequential(nn.Conv2d(features*8, features*8, 4, 2, 1), nn.ReLU())
        # Decoder
        self.up1 = UNetBlock(features*8,   features*8, down=False, dropout=True)
        self.up2 = UNetBlock(features*8*2, features*8, down=False, dropout=True)
        self.up3 = UNetBlock(features*8*2, features*8, down=False, dropout=True)
        self.up4 = UNetBlock(features*8*2, features*8, down=False)
        self.up5 = UNetBlock(features*8*2, features*4, down=False)
        self.up6 = UNetBlock(features*4*2, features*2, down=False)
        self.up7 = UNetBlock(features*2*2, features,   down=False)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features*2, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.final(torch.cat([up7, d1], 1))


# ── NDVI colormap (RdYlGn — multi-stop, matplotlib-equivalent) ────────────────
def ndvi_colormap(ndvi_norm: np.ndarray) -> np.ndarray:
    """Map [0,1] NDVI values to a proper RdYlGn RGB heatmap with 6 color stops."""
    ndvi_norm = np.clip(ndvi_norm, 0, 1)
    # 6-stop RdYlGn: deep red → red → orange → yellow → yellow-green → green
    stops = np.array([
        [0.0,  0.647, 0.059, 0.082],  # #a50f15  deep red
        [0.2,  0.906, 0.259, 0.204],  # #e74233  red
        [0.4,  0.992, 0.682, 0.318],  # #fdae51  orange
        [0.5,  1.000, 1.000, 0.600],  # #ffff99  yellow
        [0.7,  0.651, 0.851, 0.416],  # #a6d96a  yellow-green
        [0.85, 0.263, 0.671, 0.278],  # #43ab47  green
        [1.0,  0.004, 0.408, 0.216],  # #016837  dark green
    ])
    r = np.interp(ndvi_norm, stops[:, 0], stops[:, 1])
    g = np.interp(ndvi_norm, stops[:, 0], stops[:, 2])
    b = np.interp(ndvi_norm, stops[:, 0], stops[:, 3])
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def postprocess_ndvi(ndvi_arr: np.ndarray) -> np.ndarray:
    """Post-processing pipeline per Evion spec:
    1. Percentile-based histogram stretch (use full [0,1] range)
    2. Gaussian smooth (edge-preserving via PIL)
    """
    # ── Histogram stretch (2nd–98th percentile) ──────────────────────────────
    p2  = np.percentile(ndvi_arr, 2)
    p98 = np.percentile(ndvi_arr, 98)
    if (p98 - p2) > 0.01:
        ndvi_arr = (ndvi_arr - p2) / (p98 - p2)
    ndvi_arr = np.clip(ndvi_arr, 0, 1)

    # ── Gaussian smooth via PIL (simulates bilateral, removes speckle) ───────
    smooth_img = Image.fromarray((ndvi_arr * 255).astype(np.uint8), mode='L')
    smooth_img = smooth_img.filter(ImageFilter.GaussianBlur(radius=1.2))
    ndvi_arr = np.array(smooth_img).astype(np.float32) / 255.0

    return ndvi_arr


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

        # Handle different checkpoint formats gracefully
        if isinstance(checkpoint, dict):
            sd = (
                checkpoint.get("generator_state_dict")
                or checkpoint.get("model_state_dict")
                or checkpoint.get("state_dict")
                or checkpoint.get("gen")
                or checkpoint
            )
        else:
            sd = checkpoint

        gen.load_state_dict(sd, strict=False)
        gen.eval()
        model = gen
        logger.info("✅ Model loaded successfully")
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


def compute_statistics(ndvi_arr: np.ndarray) -> dict:
    return {
        "mean_ndvi":   round(float(np.mean(ndvi_arr)), 4),
        "max_ndvi":    round(float(np.max(ndvi_arr)), 4),
        "min_ndvi":    round(float(np.min(ndvi_arr)), 4),
        "healthy_pct": round(float(np.mean(ndvi_arr > 0.3)) * 100, 2),
        "stressed_pct": round(float(np.mean((ndvi_arr > 0.1) & (ndvi_arr <= 0.3))) * 100, 2),
        "barren_pct":  round(float(np.mean(ndvi_arr <= 0.1)) * 100, 2),
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
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files are accepted (JPEG/PNG).")

    contents = await file.read()
    try:
        orig = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Cannot decode uploaded file as an image.")

    orig_resized = orig.resize((INPUT_SIZE, INPUT_SIZE))
    OUTPUT_SIZE = 512  # upscale for sharper display

    # ── Inference ────────────────────────────────────────────────────────────
    if model is not None:
        tensor = transform(orig).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = model(tensor)                       # (1, 1, H, W) in [-1, 1]
        ndvi_arr = 1.0 - (pred.squeeze().cpu().numpy() + 1) / 2  # model: -1=veg, +1=barren → flip
    else:
        # Graceful mock: compute rough NDVI-like map from R/G channels
        arr = np.array(orig_resized).astype(np.float32) / 255.0
        nir_approx = arr[:, :, 1]          # green ≈ NIR proxy
        red = arr[:, :, 0]
        denom = nir_approx + red + 1e-6
        ndvi_arr = np.clip((nir_approx - red) / denom, 0, 1)

    # ── Post-processing pipeline ─────────────────────────────────────────────
    ndvi_arr = postprocess_ndvi(ndvi_arr)
    stats = compute_statistics(ndvi_arr)

    # ── Colormap heatmap ──────────────────────────────────────────────────────
    heatmap_rgb = ndvi_colormap(ndvi_arr)
    heatmap_img = Image.fromarray(heatmap_rgb)

    # ── Upscale 256 → 512 (sharper display without re-running model) ─────────
    heatmap_img = heatmap_img.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS)
    orig_display = orig_resized.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS)

    # ── HSV saturation boost ×1.4 (per pipeline spec) ────────────────────────
    from PIL import ImageEnhance
    heatmap_img = ImageEnhance.Color(heatmap_img).enhance(1.4)

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
