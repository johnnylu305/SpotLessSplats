scene="patio_high_gt_difix"
# First training
CUDA_VISIBLE_DEVICES=0 python spotless_trainer.py \
    --data_dir /home/johnny305/Documents/dfpaint/dataset/${scene}/ \
    --data_factor 8 \
    --result_dir /home/johnny305/Documents/dfpaint/dataset/results/${scene}_3dgs_composite_app_first/ \
    --loss_type l1 \
    --no-semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --app-opt \
    --disable_viewer
CUDA_VISIBLE_DEVICES=0 python spotless_trainer.py \
    --data_dir /home/johnny305/Documents/dfpaint/dataset/${scene}/ \
    --data_factor 8 \
    --result_dir /home/johnny305/Documents/dfpaint/dataset/results/${scene}_3dgs_composite_app_first/ \
    --loss_type l1 \
    --no-semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --app-opt \
    --disable_viewer \
    --ckpt /home/johnny305/Documents/dfpaint/dataset/results/${scene}_3dgs_composite_app_first/ckpts/ckpt_29999.pt 
