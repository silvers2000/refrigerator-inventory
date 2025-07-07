# verify_dataset.py
import yaml
from pathlib import Path
import sys

BASE = Path(__file__).parent

# load your data config
cfg_path = BASE / "data" / "open_images_fruits.yaml"
if not cfg_path.is_file():
    print(f"[FATAL] config not found: {cfg_path}")
    sys.exit(1)

cfg = yaml.safe_load(cfg_path.open())
nc    = cfg.get("nc")
names = cfg.get("names", [])

if nc is None:
    print("[FATAL] `nc` missing in YAML")
    sys.exit(1)
if len(names) != nc:
    print(f"[WARN] `names` length {len(names)} != nc {nc}")

def resolve_path(p):
    p = Path(p)
    # try as‐is
    if p.exists():
        return p
    # try relative to BASE
    p2 = BASE / p
    if p2.exists():
        return p2
    # try under BASE/data
    p3 = BASE / "data" / p
    if p3.exists():
        return p3
    return None

def check_split(split):
    print(f"\n=== Checking {split.upper()} ===")
    raw = cfg.get(split)
    if raw is None:
        print(f"[ERROR] `{split}` key missing in YAML")
        return

    img_dir = resolve_path(raw)
    lbl_dir = img_dir.parent / "labels" if img_dir else None

    if not img_dir or not img_dir.is_dir():
        print(f"[ERROR] images dir not found: {raw}")
        return
    if not lbl_dir or not lbl_dir.is_dir():
        print(f"[ERROR] labels dir not found: {lbl_dir}")
        return

    imgs = list(img_dir.glob("*.[jp][pn]g"))
    lbls = list(lbl_dir.glob("*.txt"))

    print(f" • {len(imgs):4d} images in {img_dir.relative_to(BASE)}")
    print(f" • {len(lbls):4d} labels in {lbl_dir.relative_to(BASE)}")

    im_bases = {f.stem for f in imgs}
    lb_bases = {f.stem for f in lbls}
    missing_lbl = im_bases - lb_bases
    missing_img = lb_bases - im_bases
    if missing_lbl:
        print(f"[WARN] {len(missing_lbl)} images without labels (e.g. {next(iter(missing_lbl))})")
    if missing_img:
        print(f"[WARN] {len(missing_img)} labels without images (e.g. {next(iter(missing_img))})")

    bad = []
    for txt in lbls:
        for L in txt.open():
            parts = L.strip().split()
            if not parts: continue
            cls = int(parts[0])
            if cls < 0 or cls >= nc:
                bad.append((txt, cls))
                break
    if bad:
        print(f"[ERROR] {len(bad)} label files with invalid class IDs:")
        for txt, cls in bad[:5]:
            print(f"   – {txt.relative_to(BASE)} → class {cls}")
    else:
        print(f" ✅ all class IDs in range 0–{nc-1}")

check_split("train")
check_split("val")
