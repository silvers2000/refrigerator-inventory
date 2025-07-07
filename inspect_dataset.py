# inspect_dataset.py
import yaml
from pathlib import Path
import sys

BASE = Path(__file__).parent
DATA_DIR = BASE / "data" / "open_images_fruits"
CFG = BASE / "data" / "open_images_fruits.yaml"

# 1) load your YAML
if not CFG.is_file():
    print(f"[FATAL] config not found: {CFG}")
    sys.exit(1)
cfg = yaml.safe_load(CFG.open())
nc = cfg.get("nc")
print(f"→ YAML says nc = {nc}\n")

# 2) discover every images/ and labels/ folder
print("🔎 scanning for images/ & labels/ directories...\n")
for kind in ("images", "labels"):
    print(f"--- found {kind}/ dirs:")
    for d in DATA_DIR.rglob(kind):
        files = list(d.glob("*"))
        print(f"  • {d.relative_to(BASE)} → {len(files)} files")
    print()

# 3) verify your declared paths actually exist & have files
def check_path(key):
    print(f"=== Checking `{key}` as declared in YAML ===")
    raw = cfg.get(key)
    if not raw:
        print("  [ERROR] missing key in YAML")
        return
    p = (BASE / raw).resolve()
    print("  declared path:", raw)
    print("  actually points to:", p)
    if not p.exists():
        print("  [ERROR] path does not exist")
        return
    imgs = list(p.glob("*.[jp][pn]g")) if "images" in key else []
    lbls = list(p.glob("*.txt"))   if "labels" in key else []
    print(f"   • images: {len(imgs)}" if "images" in key else f"   • labels: {len(lbls)}")
    # if it's images/ check for a parallel labels/ folder
    if "images" in key:
        lbl_dir = p.parent / "labels"
        if not lbl_dir.exists():
            print(f"  [WARN] no sibling labels/ at {lbl_dir.relative_to(BASE)}")
    print()

check_path("train")
check_path("val")

# 4) scan all labels/*.txt for out-of-range class IDs
print("=== Checking all label files for invalid class IDs ===")
bad = []
for txt in DATA_DIR.rglob("labels/*.txt"):
    for L in txt.open():
        parts = L.strip().split()
        if not parts: continue
        cls = int(parts[0])
        if cls < 0 or cls >= nc:
            bad.append((txt.relative_to(BASE), cls))
            break
if bad:
    print(f"  [ERROR] {len(bad)} files with bad class IDs:")
    for f,c in bad[:5]:
        print(f"    – {f} → class {c}")
else:
    print("  ✅ all class IDs in range 0–", nc-1)
