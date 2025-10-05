#!/usr/bin/env python3
import argparse
import numpy as np
import torch
from plyfile import PlyData, PlyElement

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _to_numpy(t):
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)

def save_3dgs_ply_playcanvas(
    ply_path: str,
    means3d: np.ndarray,        # (N,3)
    log_scales: np.ndarray,     # (N,3) log-space
    quats_wxyz: np.ndarray,     # (N,4) (w,x,y,z)
    opa_logits: np.ndarray,     # (N,) or (N,1)
    sh0: np.ndarray,            # (N,1,3) or (N,3)
    shN: np.ndarray             # (N,K,3)
):
    # ---- Shapes & conversions ----
    means3d   = means3d.astype(np.float32)
    scales    = log_scales #np.exp(log_scales).astype(np.float32)  # -> positive scales
    # PlayCanvas expects quaternion as (x,y,z,w)
    quats_xyzw = np.stack(
        [quats_wxyz[:,0], quats_wxyz[:,1], quats_wxyz[:,2], quats_wxyz[:,3]], axis=1
    ).astype(np.float32)
    opacity = opa_logits #_sigmoid(opa_logits).reshape(-1).astype(np.float32)

    # SH DC: (N,1,3) or (N,3) -> (N,3)
    if sh0.ndim == 3 and sh0.shape[1] == 1:
        f_dc = sh0[:,0,:].astype(np.float32)
    elif sh0.ndim == 2 and sh0.shape[1] == 3:
        f_dc = sh0.astype(np.float32)
    else:
        raise ValueError(f"Unexpected sh0 shape: {sh0.shape} (expected (N,1,3) or (N,3))")

    # SH rest: (N,K,3) -> (N, K*3)
    if shN.ndim != 3 or shN.shape[2] != 3:
        raise ValueError(f"Unexpected shN shape: {shN.shape} (expected (N,K,3))")
    N, K, _ = shN.shape
    f_rest = shN.reshape(N, K*3).astype(np.float32)

    # ---- Build PLY dtype with EXACT names & order for PlayCanvas ----
    props = [
        ('x','f4'), ('y','f4'), ('z','f4'),
        ('scale_0','f4'), ('scale_1','f4'), ('scale_2','f4'),
        ('rot_0','f4'), ('rot_1','f4'), ('rot_2','f4'), ('rot_3','f4'),
        ('opacity','f4'),
        ('f_dc_0','f4'), ('f_dc_1','f4'), ('f_dc_2','f4'),
    ] + [(f'f_rest_{i}','f4') for i in range(f_rest.shape[1])]

    vertex = np.empty(N, dtype=np.dtype(props))
    # positions
    vertex['x'], vertex['y'], vertex['z'] = -means3d[:,0], means3d[:,2], means3d[:,1]
    # scales
    vertex['scale_0'], vertex['scale_1'], vertex['scale_2'] = scales[:,0], scales[:,2], scales[:,1]
    # rotations (x,y,z,w)
    vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3'] = (
        -quats_xyzw[:,0], quats_xyzw[:,1], quats_xyzw[:,3], -quats_xyzw[:,2]
    )


    # opacity
    vertex['opacity'] = opacity
    # SH DC
    vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2'] = f_dc[:,0], f_dc[:,1], f_dc[:,2]
    # SH rest
    for i in range(f_rest.shape[1]):
        vertex[f'f_rest_{i}'] = f_rest[:, i]

    PlyData([PlyElement.describe(vertex, 'vertex')], text=False).write(ply_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pt",  help="Path to checkpoint .pt (from spotless_trainer.py)")
    ap.add_argument("ply", help="Output PLY path for PlayCanvas/SuperSplat")
    args = ap.parse_args()

    ckpt = torch.load(args.pt, map_location='cpu')
    if "splats" not in ckpt:
        raise KeyError("Expected key 'splats' in checkpoint. Got keys: " + ", ".join(ckpt.keys()))

    spl = ckpt["splats"]
    # Pull EXACT keys as defined in spotless_trainer.py
    means3d   = _to_numpy(spl["means3d"])      # (N,3)
    log_scale = _to_numpy(spl["scales"])       # (N,3) log
    quats     = _to_numpy(spl["quats"])        # (N,4) (w,x,y,z)
    opa_logit = _to_numpy(spl["opacities"])    # (N,)
    sh0       = _to_numpy(spl["sh0"])          # (N,1,3)
    shN       = _to_numpy(spl["shN"])          # (N,K,3)

    # Sanity prints (optional)
    N, K, _ = shN.shape
    L_guess = int(np.sqrt(K + 1)) - 1          # because K = (L+1)^2 - 1
    print(f"[pt_to_ply] Gaussians: {means3d.shape[0]}, SH degree ≈ {L_guess} (K={K})")
    print(f"[pt_to_ply] Writing binary PLY → {args.ply}")

    save_3dgs_ply_playcanvas(
        args.ply, means3d, log_scale, quats, opa_logit, sh0, shN
    )
    print("[pt_to_ply] Done.")

if __name__ == "__main__":
    main()

