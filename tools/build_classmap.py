# tools/build_classmap.py
import os, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "images"
OUT = IMG_DIR / "class_map.json"

def natural_key(s): return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def main():
    files = [p for p in IMG_DIR.iterdir() if p.suffix.lower() in {".png",".jpg",".jpeg",".bmp",".webp"}]
    assert len(files)==9, f"images 폴더에 9장이 있어야 합니다(현재 {len(files)})."
    files_sorted = sorted(files, key=lambda p: natural_key(p.name))
    mapping = {str(i+1): files_sorted[i].name for i in range(9)}
    data = {"root": "images", "map": mapping, "note": "sorted by filename"}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("saved:", OUT)

if __name__ == "__main__":
    main()
