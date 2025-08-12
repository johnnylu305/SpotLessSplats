scene="spot"
# training
CUDA_VISIBLE_DEVICES=1 python spotless_trainer.py \
    --data_dir /projects/MAD3D/ChengYou/CA3/${scene}/ \
    --data_factor 8 \
    --result_dir /projects/MAD3D/ChengYou/CA3/results/${scene}_gt/ \
    --loss_type robust \
    --semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --disable_viewer \
    --use_post_mask
