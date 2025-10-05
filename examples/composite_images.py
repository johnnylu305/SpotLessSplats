# fuse_ab_with_mask.py
# pip install pillow numpy

from pathlib import Path
import numpy as np
from PIL import Image

# ==== EDIT THESE PATHS ====
PATH_A   = Path("/home/johnny305/Documents/dfpaint/dataset/patio_high/images_8/")
PATH_B   = Path("/home/johnny305/Documents/dfpaint/dataset/results/patio_high_gt/renders/render_difix_noreference_29999/")
PATH_M   = Path("/home/johnny305/Documents/dfpaint/dataset/patio_high/mask/")     # mask 1=take A, 0=take B
OUT_DIR  = Path("/home/johnny305/Documents/dfpaint/dataset/results/patio_high_gt/renders/render_composite/")
# ==== OPTIONS ====
THRESH = 0.5       # mask binarization threshold in [0,1]
INVERT = False     # set True if your mask means 1=take B, 0=take A
RESIZE_TO_A = True # resize B and mask to match A's size

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def index_by_stem(folder: Path):
    return {p.stem: p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS}

def load_image_rgb(path: Path, size=None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, resample=Image.BILINEAR)
    return np.array(img, dtype=np.uint8)

def load_mask(path: Path, size=None, thresh: float = 0.5, invert: bool = False) -> np.ndarray:
    img = Image.open(path).convert("L")
    if size is not None and img.size != size:
        img = img.resize(size, resample=Image.NEAREST)
    m = np.array(img, dtype=np.float32)
    if m.max() > 1.0:   # handle 0..255 masks
        m = m / 255.0
    m = (m >= thresh).astype(np.float32)  # binarize to {0,1}
    if invert:
        m = 1.0 - m
    return m[..., None]  # HxW -> HxWx1 for broadcasting

def fuse(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> np.ndarray:
    # a,b: uint8 HxWx3, m: float32 HxWx1 in {0,1}
    out = a.astype(np.float32) * m + b.astype(np.float32) * (1.0 - m)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)

def main():
    assert PATH_A.is_dir() and PATH_B.is_dir() and PATH_M.is_dir(), "Input folders must exist"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    a_idx = index_by_stem(PATH_A)
    b_idx = index_by_stem(PATH_B)
    m_idx = index_by_stem(PATH_M)

    done = skipped = 0
    for stem in sorted(a_idx.keys()):
        a_path = a_idx[stem]
        b_path = b_idx.get("train_"+stem)
        m_path = m_idx.get("train_"+stem)


        if b_path is None or m_path is None:
            print(f"[skip] {stem}: B={'OK' if b_path else 'MISSING'} mask={'OK' if m_path else 'MISSING'}")
            skipped += 1
            continue

        a_img = Image.open(a_path).convert("RGB")
        A = np.array(a_img, dtype=np.uint8)

        if RESIZE_TO_A:
            B = load_image_rgb(b_path, size=a_img.size)
            M = load_mask(m_path, size=a_img.size, thresh=THRESH, invert=INVERT)
        else:
            b_img = Image.open(b_path).convert("RGB")
            m_img = Image.open(m_path).convert("L")
            if b_img.size != a_img.size or m_img.size != a_img.size:
                print(f"[skip] {stem}: size mismatch A{a_img.size} B{b_img.size} M{m_img.size}")
                skipped += 1
                continue
            B = np.array(b_img, dtype=np.uint8)
            M = load_mask(m_path, size=None, thresh=THRESH, invert=INVERT)

        fused = fuse(A, B, M)
        out_path = OUT_DIR / f"{stem}.png"
        Image.fromarray(fused).save(out_path)
        done += 1

    print(f"Done: {done}, skipped: {skipped}, out: {OUT_DIR}")

if __name__ == "__main__":
    main()

