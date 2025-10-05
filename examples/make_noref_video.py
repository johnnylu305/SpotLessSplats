import os
import cv2
import numpy as np
from glob import glob

# ---------- HARD-CODED PATHS ----------
ORG_DIR = "/home/johnny305/Documents/dfpaint/dataset/results/patio_high_first/renders/data_29999/"
SLS_DIR = "/home/johnny305/Documents/dfpaint/dataset/results/patio_high_first/renders/render_29999/"
DIFIX_DIR = "/home/johnny305/Documents/dfpaint/dataset/results/patio_high_first/renders/render_difix_noreference_29999/"

OUT_DIR = "/home/johnny305/Documents/dfpaint/dataset/results/patio_high_first/renders/"
OUT_FRAMES = os.path.join(OUT_DIR, "frames")
OUT_VIDEO = os.path.join(OUT_DIR, "video.mp4")
FPS = 1  # change if needed
# --------------------------------------

os.makedirs(OUT_FRAMES, exist_ok=True)

def read_img(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read {path}")
    return img

def to_rgb(img):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def abs_err(a, b):
    # a, b are BGR uint8 images, same size
    diff = cv2.absdiff(a, b)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)  # consistent scale (0..255)
    return cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)

# Collect filenames present in all three folders (by basename)
org_files = {os.path.basename(p): p for p in sorted(glob(os.path.join(ORG_DIR, "*")))}
sls_files = {os.path.basename(p): p for p in sorted(glob(os.path.join(SLS_DIR, "*")))}
dif_files = {os.path.basename(p): p for p in sorted(glob(os.path.join(DIFIX_DIR, "*")))}

common_names = sorted(set(org_files) & set(sls_files) & set(dif_files))
if not common_names:
    raise RuntimeError("No common filenames across the three folders.")

writer = None
frame_paths = []

for i, name in enumerate(common_names):
    org = read_img(org_files[name])
    sls = read_img(sls_files[name])
    dif = read_img(dif_files[name])

    h, w = org.shape[:2]

    # Resize SLS and Difix to Org’s size to align
    sls = cv2.resize(sls, (w, h), interpolation=cv2.INTER_AREA)
    dif = cv2.resize(dif, (w, h), interpolation=cv2.INTER_AREA)

    # Compute errors
    err_org_dif = abs_err(org, dif)
    err_sls_dif = abs_err(sls, dif)
    err_org_sls = abs_err(org, sls)  # <-- new bottom-right tile

    # Compose 2x3 grid
    top = np.hstack([to_rgb(org), to_rgb(sls), to_rgb(dif)])
    bottom = np.hstack([err_org_dif, err_org_sls, err_sls_dif])  # use Org-SLS
    grid = np.vstack([top, bottom])

    # Optional: draw small labels
    def put_label(im, text, x, y):
        cv2.putText(im, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

    put_label(grid, "Org", 10, 30)
    put_label(grid, "SLS", w + 10, 30)
    put_label(grid, "Difix", 2*w + 10, 30)
    put_label(grid, "|Org - Difix|", 10, h + 30)
    put_label(grid, "|Org - SLS|", w + 10, h + 30)
    put_label(grid, "|SLS - Difix|", 2*w + 10, h + 30)  # <-- updated label

    out_path = os.path.join(OUT_FRAMES, f"frame_{i:05d}.png")
    cv2.imwrite(out_path, grid)
    frame_paths.append(out_path)

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        gh, gw = grid.shape[:2]
        os.makedirs(OUT_DIR, exist_ok=True)
        writer = cv2.VideoWriter(OUT_VIDEO, fourcc, FPS, (gw, gh))

    writer.write(grid)

if writer is not None:
    writer.release()

print(f"Saved {len(frame_paths)} frames to: {OUT_FRAMES}")
print(f"Saved video to: {OUT_VIDEO}")

