"""
Run once to inspect the .pth checkpoint structure.
Usage: python inspect_model.py
"""
import torch, sys, os

path = os.path.join(os.path.dirname(__file__), "..", "best_model (1).pth")
print(f"Loading: {path}\n")

ckpt = torch.load(path, map_location="cpu")
print("Top-level type:", type(ckpt))

if isinstance(ckpt, dict):
    print("Top-level keys:", list(ckpt.keys()))
    for k, v in ckpt.items():
        if isinstance(v, dict):
            keys = list(v.keys())
            print(f"\n[{k}] — {len(keys)} params")
            for pk in keys[:8]:
                print(f"  {pk}: {list(v[pk].shape)}")
            if len(keys) > 8:
                print(f"  ... and {len(keys)-8} more")
        elif hasattr(v, 'shape'):
            print(f"[{k}]: tensor {list(v.shape)}")
        else:
            print(f"[{k}]: {v}")
else:
    keys = list(ckpt.keys())
    print(f"Direct OrderedDict — {len(keys)} params")
    for pk in keys[:12]:
        print(f"  {pk}: {list(ckpt[pk].shape)}")
    if len(keys) > 12:
        print(f"  ... and {len(keys)-12} more")
