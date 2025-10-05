import os
from PIL import Image

# Input and output folders
input_dir = "../../dataset/patio_high/gt/gt_vis/mask"
output_dir = "../../dataset/patio_high/gt/gt_vis/mask_8"

os.makedirs(output_dir, exist_ok=True)

# Loop over all files in mask/
for fname in os.listdir(input_dir):
    in_path = os.path.join(input_dir, fname)
    out_path = os.path.join(output_dir, fname)

    # Skip non-image files
    if not (fname.lower().endswith(".png") or fname.lower().endswith(".jpg") or fname.lower().endswith(".jpeg")):
        continue

    # Open image
    img = Image.open(in_path)

    # Compute new size (8x smaller)
    new_size = (img.width // 8, img.height // 8)

    # Resize with nearest neighbor (preserves mask labels)
    img_small = img.resize(new_size, Image.NEAREST)

    # Save to mask_8
    img_small.save(out_path)

    print(f"Saved: {out_path}")

