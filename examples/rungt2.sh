scene="patio_high"
# training
CUDA_VISIBLE_DEVICES=0 python3 spotless_trainer.py \
    --data_dir /home/johnny305/Documents/dfpaint/dataset/${scene}/ \
    --data_factor 8 \
    --result_dir /home/johnny305/Documents/dfpaint/dataset/results/${scene}_gt4/ \
    --loss_type robust \
    --semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --disable_viewer \
    --use_post_mask
CUDA_VISIBLE_DEVICES=0 python spotless_trainer.py \
    --data_dir /home/johnny305/Documents/dfpaint/dataset/${scene}/ \
    --data_factor 8 \
    --result_dir /home/johnny305/Documents/dfpaint/dataset/results/${scene}_gt4/ \
    --loss_type robust \
    --semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --disable_viewer \
    --ckpt /home/johnny305/Documents/dfpaint/dataset/results/${scene}_gt2/ckpts/ckpt_29999.pt
