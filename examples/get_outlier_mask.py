import os
import cv2
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch.nn.functional as F
from PIL import Image

# ----------- Config -----------
DATA_FOLDER = "/projects/MAD3D/ChengYou/CA3/results_test/renders/data"
RENDER_FOLDER = "/projects/MAD3D/ChengYou/CA3/results_test/renders/render"
# vit_h > vit_l > vit_b
SAM_CHECKPOINT = "/home/chlu/Downloads/sam_vit_h_4b8939.pth"
SIM_THRESHOLD = 0.7
RATE = 1 
# ------------------------------

# Load SAM model
sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT).to("cuda")
# mask for entire image
#mask_generator = SamAutomaticMaskGenerator(sam)
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
)



# Load DINOv2
# vit_g > vit_l > vit_b > vit_s
# TODO:
# we can try register one which is better for segmentation task
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to("cuda").eval()

# TODO: if one pixel belong to multiple mask, can we try ensemble?
# one pixel can only belong to one mask
def make_exclusive_mask_map(masks, h, w, priority="score"):
    # 0 means the pixel does not belong to any mask
    label_map = np.zeros((h, w), dtype=np.int32)
    if priority == "score":
        sorted_masks = sorted(enumerate(masks), key=lambda x: x[1].get("predicted_iou", 0), reverse=True)
    elif priority == "area":
        sorted_masks = sorted(enumerate(masks), key=lambda x: x[1]["segmentation"].sum(), reverse=True)
    else:
        sorted_masks = list(enumerate(masks))

    for idx, m in sorted_masks:
        seg = m["segmentation"]
        update_mask = (seg == 1) & (label_map == 0)
        label_map[update_mask] = idx + 1  # label 0 is background

    return label_map

# show sam mask one by one
def show_each_sam_mask(img_rgb, masks, title_prefix="Mask"):
    for i, m in enumerate(masks):
        mask = m["segmentation"].astype(bool)
        plt.figure(figsize=(8, 6))
        plt.imshow(img_rgb)
        plt.imshow(np.ma.masked_where(~mask, mask), cmap='jet', alpha=0.5)
        plt.title(f"{title_prefix} #{i} — area: {mask.sum()} pixels")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

# show sam masks together
# non-mask pixels are not included
def show_sam_masks(img_rgb, masks, title="SAM Masks"):
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    for idx, m in enumerate(masks):
        mask = m["segmentation"].astype(bool)
        plt.imshow(np.ma.masked_where(~mask, mask), cmap='jet', alpha=0.7)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour = contour.squeeze()
            if contour.ndim == 2 and len(contour) > 2:
                plt.plot(contour[:, 0], contour[:, 1], color='white', linewidth=1.0)
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.show()

# show similarity heatmap
# noted, 0 may mean the pixel does not belong to any mask
def show_similarity_heatmap(sim_map, title="Cosine Similarity Heatmap"):
    plt.figure(figsize=(8, 6))
    plt.imshow(sim_map, cmap='jet', vmin=0, vmax=1)
    plt.colorbar(label='Cosine Similarity')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# extract dinov2 features per patch
def get_patch_features(img_rgb, model):
    h, w = img_rgb.shape[:2]
    # resize and make it 14 multiple
    resize_h = (h // RATE) // 14 * 14
    resize_w = (w // RATE) // 14 * 14
    transform = transforms.Compose([
        transforms.Resize((resize_h, resize_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    pil_img = Image.fromarray(img_rgb)
    img_tensor = transform(pil_img).unsqueeze(0).to("cuda")
    with torch.no_grad():
        # extract normalized feature per patch
        feats = model.forward_features(img_tensor)["x_norm_patchtokens"]
    grid_h, grid_w = resize_h // 14, resize_w // 14
    feats = feats.view(1, grid_h, grid_w, -1).permute(0, 3, 1, 2)
    feats_upsampled = F.interpolate(feats, size=(resize_h, resize_w), mode="bilinear", align_corners=False)
    return feats_upsampled.squeeze(0), resize_h, resize_w

# save sam segmentation
def save_sam_masks(img_rgb, masks, save_path):
    overlay = img_rgb.copy()
    for m in masks:
        color = [random.randint(0, 255) for _ in range(3)]
        #mask = (label_map == (idx + 1)).astype(np.uint8)
        color_mask = np.zeros_like(img_rgb, dtype=np.uint8)
        for c in range(3):
            color_mask[:, :, c] = color[c]
        overlay = np.where(m["segmentation"][:, :, None], overlay * 0.1 + color_mask * 0.9, overlay)
        contours, _ = cv2.findContours(m["segmentation"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)
    vis_bgr = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, vis_bgr)

# save outlier mask with blue car in rgb image
def save_outlier_vis(img_rgb, masks, outlier_mask, save_path):
    vis = img_rgb.copy()
    for m in masks:
        mask = m["segmentation"].astype(np.uint8)
        if np.any((outlier_mask > 0) & (mask > 0)):
            blue = [0, 0, 255]
            for c in range(3):
                vis[:, :, c] = np.where(mask == 1, blue[c], vis[:, :, c])
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (255, 255, 255), 1)
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, vis_bgr)

# save similarity map as heatmap
def save_similarity_heatmap(sim_map, save_path):
    sim_map_clipped = np.clip(sim_map, 0.0, 1.0)
    sim_uint8 = (sim_map_clipped * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(sim_uint8, cv2.COLORMAP_JET)
    cv2.imwrite(save_path, heatmap)

# average pooling with sam mask 
def compute_mean_feature(per_pixel_feat, mask, resize_w, resize_h):
    mask_resized = cv2.resize(mask.astype(np.uint8), (resize_w, resize_h), interpolation=cv2.INTER_NEAREST)
    mask_tensor = torch.from_numpy(mask_resized).to(per_pixel_feat.device).float()
    mask_tensor = mask_tensor.unsqueeze(0)
    masked_feat = per_pixel_feat * mask_tensor
    feat_sum = masked_feat.sum(dim=(1, 2))
    mask_area = mask_tensor.sum()
    return None if mask_area == 0 else feat_sum / mask_area

def cosine_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

image_files = sorted([
    f for f in os.listdir(DATA_FOLDER)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    and not any(substr in f.lower() for substr in ['sam', 'outlier', 'similarity'])
])

for fname in tqdm(image_files, desc="Processing"):
    data_path = os.path.join(DATA_FOLDER, fname)
    render_path = os.path.join(RENDER_FOLDER, fname)
    if not os.path.exists(render_path):
        continue

    img_data = cv2.imread(data_path)
    img_render = cv2.imread(render_path)
    img_data_rgb = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    img_render_rgb = cv2.cvtColor(img_render, cv2.COLOR_BGR2RGB)

    masks = mask_generator.generate(img_data_rgb)

    h, w = img_data.shape[:2]
    outlier_mask = np.zeros((h, w), dtype=np.uint8)
    sim_map = np.zeros((h, w), dtype=np.float32)
    
    # extract dinov2 feature per patch
    feat_data_map, resize_h, resize_w = get_patch_features(img_data_rgb, dinov2)
    feat_render_map, _, _ = get_patch_features(img_render_rgb, dinov2)

    # one mask per pixel
    label_map = make_exclusive_mask_map(masks, h, w, priority="score")

    for idx, m in enumerate(masks):
        # 0 for non-mask pixel
        mask_id = idx + 1
        seg = (label_map == mask_id).astype(np.uint8)
        if seg.sum() == 0:
            continue

        # average pooling per mask
        feat_data = compute_mean_feature(feat_data_map, seg, resize_w, resize_h)
        feat_render = compute_mean_feature(feat_render_map, seg, resize_w, resize_h)
        if feat_data is None or feat_render is None:
            continue

        # find outlier
        sim = cosine_sim(feat_data, feat_render)
        if sim < SIM_THRESHOLD:
            outlier_mask[seg > 0] = 255

        sim_map[seg > 0] = sim

    base = os.path.splitext(fname)[0]
    cv2.imwrite(os.path.join(DATA_FOLDER, f"{base}_outlier.png"), outlier_mask)
    save_sam_masks(img_data_rgb, [{"segmentation": (label_map == (i+1))} for i in range(len(masks))], os.path.join(DATA_FOLDER, f"{base}_sam_viz.png"))
    save_outlier_vis(img_data_rgb, [{"segmentation": (label_map == (i+1))} for i in range(len(masks))], outlier_mask, os.path.join(DATA_FOLDER, f"{base}_outlier_viz.png"))
    save_similarity_heatmap(sim_map, os.path.join(DATA_FOLDER, f"{base}_similarity_heatmap.png"))
    
    #show_each_sam_mask(img_data_rgb, [{"segmentation": (label_map == (i+1))} for i in range(len(masks))])
    #show_similarity_heatmap(sim_map, title=f"Similarity: {fname}")
    #show_sam_masks(img_data_rgb, [{"segmentation": (label_map == (i+1))} for i in range(len(masks))], title="SAM Masks")

