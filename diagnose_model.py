"""
Diagnostic: Verify the fixed architecture loads correctly with strict=True
and produces sensible outputs matching the Kaggle evaluation notebook.
"""
import torch
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from api_server import Generator, to_ndvi_image, ndvi_from_rgb

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "best_model (1).pth")

# ── Step 1: Load model with strict=True ──────────────────────────────────────
print(f"Loading model from: {MODEL_PATH}")
checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
gen = Generator()

if isinstance(checkpoint, dict):
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    epoch = checkpoint.get("epoch", "?")
    g_loss = checkpoint.get("g_loss", "?")
    print(f"Epoch: {epoch}, G_loss: {g_loss}")
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

try:
    gen.load_state_dict(sd, strict=True)
    print("✅ strict=True load PASSED — all checkpoint weights matched the architecture!")
except RuntimeError as e:
    print(f"❌ strict=True load FAILED: {e}")
    print("Falling back to strict=False for diagnostics...")
    gen.load_state_dict(sd, strict=False)

gen.eval()
print()

# ── Step 2: Verify parameter matching ────────────────────────────────────────
model_keys = set(gen.state_dict().keys())
ckpt_keys = set(sd.keys())
missing = model_keys - ckpt_keys
unexpected = ckpt_keys - model_keys
print(f"Model parameters: {len(model_keys)}")
print(f"Checkpoint parameters: {len(ckpt_keys)}")
print(f"Missing from checkpoint: {len(missing)}")
print(f"Unexpected in checkpoint: {len(unexpected)}")
if missing:
    print(f"  Missing: {sorted(list(missing))[:10]}...")
if unexpected:
    print(f"  Unexpected: {sorted(list(unexpected))[:10]}...")
print()

# ── Step 3: Test with controlled inputs ──────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def test_image(name, img):
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        pred = gen(tensor)  # (1, 3, 256, 256)
    
    # Kaggle-style denormalization
    pred_rgb = to_ndvi_image(pred)  # (256, 256, 3) in [0, 1]
    ndvi_single = ndvi_from_rgb(pred_rgb)  # single-channel proxy

    print(f"=== {name} ===")
    print(f"  Pred shape: {pred.shape}")
    print(f"  Raw output  → min={pred.min():.4f}, max={pred.max():.4f}, mean={pred.mean():.4f}")
    print(f"  RGB [0,1]   → min={pred_rgb.min():.4f}, max={pred_rgb.max():.4f}, mean={pred_rgb.mean():.4f}")
    print(f"  NDVI proxy  → min={ndvi_single.min():.4f}, max={ndvi_single.max():.4f}, mean={ndvi_single.mean():.4f}")
    print()
    return pred_rgb, ndvi_single

# Pure green (should be vegetation)
green = Image.new("RGB", (256, 256), (34, 139, 34))
_, ndvi_green = test_image("PURE GREEN (expect HIGH NDVI)", green)

# Pure brown (should be barren)
brown = Image.new("RGB", (256, 256), (139, 119, 101))
_, ndvi_brown = test_image("PURE BROWN/SOIL (expect LOW NDVI)", brown)

# Bright vegetation
bright = Image.new("RGB", (256, 256), (0, 200, 0))
_, ndvi_bright = test_image("BRIGHT GREEN (expect HIGHEST NDVI)", bright)

# Half green, half brown
half = Image.new("RGB", (256, 256), (139, 119, 101))
draw = ImageDraw.Draw(half)
draw.rectangle([0, 0, 128, 256], fill=(34, 139, 34))
pred_half, ndvi_half = test_image("HALF GREEN / HALF BROWN", half)

# Analyze spatial correctness
left = ndvi_half[:, :128].mean()
right = ndvi_half[:, 128:].mean()
print("SPATIAL ANALYSIS (half image):")
print(f"  Green side NDVI mean: {left:.4f}")
print(f"  Brown side NDVI mean: {right:.4f}")
if left > right:
    print("  ✅ Green side has HIGHER NDVI → correct behavior!")
else:
    print("  ⚠️  Green side has LOWER NDVI → may need investigation")
print()

# Save a sample output for visual inspection
try:
    out_rgb = (pred_half * 255).astype(np.uint8)
    out_img = Image.fromarray(out_rgb)
    out_path = os.path.join(os.path.dirname(__file__), "diagnostic_output.png")
    out_img.save(out_path)
    print(f"✅ Diagnostic output saved to: {out_path}")
except Exception as e:
    print(f"Could not save diagnostic image: {e}")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("DIAGNOSTIC SUMMARY")
print("="*60)
print(f"  Architecture: 3-channel output (Kaggle-aligned)")
print(f"  Weight loading: {'strict=True PASSED' if len(missing) == 0 and len(unexpected) == 0 else 'ISSUES DETECTED'}")
print(f"  Green NDVI mean: {ndvi_green.mean():.4f}")
print(f"  Brown NDVI mean: {ndvi_brown.mean():.4f}")
print(f"  Spatial correctness: {'✅ PASS' if left > right else '⚠️ CHECK'}")
