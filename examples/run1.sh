scene="android"
# First training
CUDA_VISIBLE_DEVICES=1 python spotless_trainer.py \
    --data_dir /projects/MAD3D/ChengYou/CA3/${scene}/ \
    --data_factor 8 \
    --result_dir /projects/MAD3D/ChengYou/CA3/results/${scene}_first/ \
    --loss_type robust \
    --semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --disable_viewer
CUDA_VISIBLE_DEVICES=1 python spotless_trainer.py \
    --data_dir /projects/MAD3D/ChengYou/CA3/${scene}/ \
    --data_factor 8 \
    --result_dir /projects/MAD3D/ChengYou/CA3/results/${scene}_first/ \
    --loss_type robust \
    --semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --disable_viewer \
    --ckpt ../../../results/${scene}_first/ckpts/ckpt_29999.pt 
# Copy mask
rm -rf /projects/MAD3D/ChengYou/CA3/${scene}/mask
cp -r /projects/MAD3D/ChengYou/CA3/results/${scene}_first/renders/mask_29999 /projects/MAD3D/ChengYou/CA3/${scene}/mask
# Second training
CUDA_VISIBLE_DEVICES=1 python spotless_trainer.py \
    --data_dir /projects/MAD3D/ChengYou/CA3/${scene}/ \
    --data_factor 8 \
    --result_dir /projects/MAD3D/ChengYou/CA3/results/${scene}_second/ \
    --loss_type robust \
    --semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --disable_viewer \
    --use_post_mask


scene="corner"
# First training
CUDA_VISIBLE_DEVICES=1 python spotless_trainer.py \
    --data_dir /projects/MAD3D/ChengYou/CA3/${scene}/ \
    --data_factor 8 \
    --result_dir /projects/MAD3D/ChengYou/CA3/results/${scene}_first/ \
    --loss_type robust \
    --semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --disable_viewer
CUDA_VISIBLE_DEVICES=1 python spotless_trainer.py \
    --data_dir /projects/MAD3D/ChengYou/CA3/${scene}/ \
    --data_factor 8 \
    --result_dir /projects/MAD3D/ChengYou/CA3/results/${scene}_first/ \
    --loss_type robust \
    --semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --disable_viewer \
    --ckpt ../../../results/${scene}_first/ckpts/ckpt_29999.pt 
# Copy mask
rm -rf /projects/MAD3D/ChengYou/CA3/${scene}/mask
cp -r /projects/MAD3D/ChengYou/CA3/results/${scene}_first/renders/mask_29999 /projects/MAD3D/ChengYou/CA3/${scene}/mask
# Second training
CUDA_VISIBLE_DEVICES=1 python spotless_trainer.py \
    --data_dir /projects/MAD3D/ChengYou/CA3/${scene}/ \
    --data_factor 8 \
    --result_dir /projects/MAD3D/ChengYou/CA3/results/${scene}_second/ \
    --loss_type robust \
    --semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --disable_viewer \
    --use_post_mask

scene="crab2"
# First training
CUDA_VISIBLE_DEVICES=1 python spotless_trainer.py \
    --data_dir /projects/MAD3D/ChengYou/CA3/${scene}/ \
    --data_factor 8 \
    --result_dir /projects/MAD3D/ChengYou/CA3/results/${scene}_first/ \
    --loss_type robust \
    --semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --disable_viewer
CUDA_VISIBLE_DEVICES=1 python spotless_trainer.py \
    --data_dir /projects/MAD3D/ChengYou/CA3/${scene}/ \
    --data_factor 8 \
    --result_dir /projects/MAD3D/ChengYou/CA3/results/${scene}_first/ \
    --loss_type robust \
    --semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --disable_viewer \
    --ckpt ../../../results/${scene}_first/ckpts/ckpt_29999.pt 
# Copy mask
rm -rf /projects/MAD3D/ChengYou/CA3/${scene}/mask
cp -r /projects/MAD3D/ChengYou/CA3/results/${scene}_first/renders/mask_29999 /projects/MAD3D/ChengYou/CA3/${scene}/mask
# Second training
CUDA_VISIBLE_DEVICES=1 python spotless_trainer.py \
    --data_dir /projects/MAD3D/ChengYou/CA3/${scene}/ \
    --data_factor 8 \
    --result_dir /projects/MAD3D/ChengYou/CA3/results/${scene}_second/ \
    --loss_type robust \
    --semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --disable_viewer \
    --use_post_mask


scene="fountain"
# First training
CUDA_VISIBLE_DEVICES=1 python spotless_trainer.py \
    --data_dir /projects/MAD3D/ChengYou/CA3/${scene}/ \
    --data_factor 8 \
    --result_dir /projects/MAD3D/ChengYou/CA3/results/${scene}_first/ \
    --loss_type robust \
    --semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --disable_viewer
CUDA_VISIBLE_DEVICES=1 python spotless_trainer.py \
    --data_dir /projects/MAD3D/ChengYou/CA3/${scene}/ \
    --data_factor 8 \
    --result_dir /projects/MAD3D/ChengYou/CA3/results/${scene}_first/ \
    --loss_type robust \
    --semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --disable_viewer \
    --ckpt ../../../results/${scene}_first/ckpts/ckpt_29999.pt 
# Copy mask
rm -rf /projects/MAD3D/ChengYou/CA3/${scene}/mask
cp -r /projects/MAD3D/ChengYou/CA3/results/${scene}_first/renders/mask_29999 /projects/MAD3D/ChengYou/CA3/${scene}/mask
# Second training
CUDA_VISIBLE_DEVICES=1 python spotless_trainer.py \
    --data_dir /projects/MAD3D/ChengYou/CA3/${scene}/ \
    --data_factor 8 \
    --result_dir /projects/MAD3D/ChengYou/CA3/results/${scene}_second/ \
    --loss_type robust \
    --semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --disable_viewer \
    --use_post_mask


scene="mountain"
# First training
CUDA_VISIBLE_DEVICES=1 python spotless_trainer.py \
    --data_dir /projects/MAD3D/ChengYou/CA3/${scene}/ \
    --data_factor 8 \
    --result_dir /projects/MAD3D/ChengYou/CA3/results/${scene}_first/ \
    --loss_type robust \
    --semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --disable_viewer
CUDA_VISIBLE_DEVICES=1 python spotless_trainer.py \
    --data_dir /projects/MAD3D/ChengYou/CA3/${scene}/ \
    --data_factor 8 \
    --result_dir /projects/MAD3D/ChengYou/CA3/results/${scene}_first/ \
    --loss_type robust \
    --semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --disable_viewer \
    --ckpt ../../../results/${scene}_first/ckpts/ckpt_29999.pt 
# Copy mask
rm -rf /projects/MAD3D/ChengYou/CA3/${scene}/mask
cp -r /projects/MAD3D/ChengYou/CA3/results/${scene}_first/renders/mask_29999 /projects/MAD3D/ChengYou/CA3/${scene}/mask
# Second training
CUDA_VISIBLE_DEVICES=1 python spotless_trainer.py \
    --data_dir /projects/MAD3D/ChengYou/CA3/${scene}/ \
    --data_factor 8 \
    --result_dir /projects/MAD3D/ChengYou/CA3/results/${scene}_second/ \
    --loss_type robust \
    --semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --disable_viewer \
    --use_post_mask

scene="patio"
# First training
CUDA_VISIBLE_DEVICES=1 python spotless_trainer.py \
    --data_dir /projects/MAD3D/ChengYou/CA3/${scene}/ \
    --data_factor 8 \
    --result_dir /projects/MAD3D/ChengYou/CA3/results/${scene}_first/ \
    --loss_type robust \
    --semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --disable_viewer
CUDA_VISIBLE_DEVICES=1 python spotless_trainer.py \
    --data_dir /projects/MAD3D/ChengYou/CA3/${scene}/ \
    --data_factor 8 \
    --result_dir /projects/MAD3D/ChengYou/CA3/results/${scene}_first/ \
    --loss_type robust \
    --semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --disable_viewer \
    --ckpt ../../../results/${scene}_first/ckpts/ckpt_29999.pt 
# Copy mask
rm -rf /projects/MAD3D/ChengYou/CA3/${scene}/mask
cp -r /projects/MAD3D/ChengYou/CA3/results/${scene}_first/renders/mask_29999 /projects/MAD3D/ChengYou/CA3/${scene}/mask
# Second training
CUDA_VISIBLE_DEVICES=1 python spotless_trainer.py \
    --data_dir /projects/MAD3D/ChengYou/CA3/${scene}/ \
    --data_factor 8 \
    --result_dir /projects/MAD3D/ChengYou/CA3/results/${scene}_second/ \
    --loss_type robust \
    --semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --disable_viewer \
    --use_post_mask


scene="patio_high"
# First training
CUDA_VISIBLE_DEVICES=1 python spotless_trainer.py \
    --data_dir /projects/MAD3D/ChengYou/CA3/${scene}/ \
    --data_factor 8 \
    --result_dir /projects/MAD3D/ChengYou/CA3/results/${scene}_first/ \
    --loss_type robust \
    --semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --disable_viewer
CUDA_VISIBLE_DEVICES=1 python spotless_trainer.py \
    --data_dir /projects/MAD3D/ChengYou/CA3/${scene}/ \
    --data_factor 8 \
    --result_dir /projects/MAD3D/ChengYou/CA3/results/${scene}_first/ \
    --loss_type robust \
    --semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --disable_viewer \
    --ckpt ../../../results/${scene}_first/ckpts/ckpt_29999.pt 
# Copy mask
rm -rf /projects/MAD3D/ChengYou/CA3/${scene}/mask
cp -r /projects/MAD3D/ChengYou/CA3/results/${scene}_first/renders/mask_29999 /projects/MAD3D/ChengYou/CA3/${scene}/mask
# Second training
CUDA_VISIBLE_DEVICES=1 python spotless_trainer.py \
    --data_dir /projects/MAD3D/ChengYou/CA3/${scene}/ \
    --data_factor 8 \
    --result_dir /projects/MAD3D/ChengYou/CA3/results/${scene}_second/ \
    --loss_type robust \
    --semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --disable_viewer \
    --use_post_mask

scene="spot"
# First training
CUDA_VISIBLE_DEVICES=1 python spotless_trainer.py \
    --data_dir /projects/MAD3D/ChengYou/CA3/${scene}/ \
    --data_factor 8 \
    --result_dir /projects/MAD3D/ChengYou/CA3/results/${scene}_first/ \
    --loss_type robust \
    --semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --disable_viewer
CUDA_VISIBLE_DEVICES=1 python spotless_trainer.py \
    --data_dir /projects/MAD3D/ChengYou/CA3/${scene}/ \
    --data_factor 8 \
    --result_dir /projects/MAD3D/ChengYou/CA3/results/${scene}_first/ \
    --loss_type robust \
    --semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --disable_viewer \
    --ckpt ../../../results/${scene}_first/ckpts/ckpt_29999.pt 
# Copy mask
rm -rf /projects/MAD3D/ChengYou/CA3/${scene}/mask
cp -r /projects/MAD3D/ChengYou/CA3/results/${scene}_first/renders/mask_29999 /projects/MAD3D/ChengYou/CA3/${scene}/mask
# Second training
CUDA_VISIBLE_DEVICES=1 python spotless_trainer.py \
    --data_dir /projects/MAD3D/ChengYou/CA3/${scene}/ \
    --data_factor 8 \
    --result_dir /projects/MAD3D/ChengYou/CA3/results/${scene}_second/ \
    --loss_type robust \
    --semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --disable_viewer \
    --use_post_mask


scene="statue"
# First training
CUDA_VISIBLE_DEVICES=1 python spotless_trainer.py \
    --data_dir /projects/MAD3D/ChengYou/CA3/${scene}/ \
    --data_factor 8 \
    --result_dir /projects/MAD3D/ChengYou/CA3/results/${scene}_first/ \
    --loss_type robust \
    --semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --disable_viewer
CUDA_VISIBLE_DEVICES=1 python spotless_trainer.py \
    --data_dir /projects/MAD3D/ChengYou/CA3/${scene}/ \
    --data_factor 8 \
    --result_dir /projects/MAD3D/ChengYou/CA3/results/${scene}_first/ \
    --loss_type robust \
    --semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --disable_viewer \
    --ckpt ../../../results/${scene}_first/ckpts/ckpt_29999.pt 
# Copy mask
rm -rf /projects/MAD3D/ChengYou/CA3/${scene}/mask
cp -r /projects/MAD3D/ChengYou/CA3/results/${scene}_first/renders/mask_29999 /projects/MAD3D/ChengYou/CA3/${scene}/mask
# Second training
CUDA_VISIBLE_DEVICES=1 python spotless_trainer.py \
    --data_dir /projects/MAD3D/ChengYou/CA3/${scene}/ \
    --data_factor 8 \
    --result_dir /projects/MAD3D/ChengYou/CA3/results/${scene}_second/ \
    --loss_type robust \
    --semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --disable_viewer \
    --use_post_mask


scene="yoda"
# First training
CUDA_VISIBLE_DEVICES=1 python spotless_trainer.py \
    --data_dir /projects/MAD3D/ChengYou/CA3/${scene}/ \
    --data_factor 8 \
    --result_dir /projects/MAD3D/ChengYou/CA3/results/${scene}_first/ \
    --loss_type robust \
    --semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --disable_viewer
CUDA_VISIBLE_DEVICES=1 python spotless_trainer.py \
    --data_dir /projects/MAD3D/ChengYou/CA3/${scene}/ \
    --data_factor 8 \
    --result_dir /projects/MAD3D/ChengYou/CA3/results/${scene}_first/ \
    --loss_type robust \
    --semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --disable_viewer \
    --ckpt ../../../results/${scene}_first/ckpts/ckpt_29999.pt 
# Copy mask
rm -rf /projects/MAD3D/ChengYou/CA3/${scene}/mask
cp -r /projects/MAD3D/ChengYou/CA3/results/${scene}_first/renders/mask_29999 /projects/MAD3D/ChengYou/CA3/${scene}/mask
# Second training
CUDA_VISIBLE_DEVICES=1 python spotless_trainer.py \
    --data_dir /projects/MAD3D/ChengYou/CA3/${scene}/ \
    --data_factor 8 \
    --result_dir /projects/MAD3D/ChengYou/CA3/results/${scene}_second/ \
    --loss_type robust \
    --semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --disable_viewer \
    --use_post_mask
