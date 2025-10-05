scene="patio_high_gt_difix"
# First training
CUDA_VISIBLE_DEVICES=0 python spotless_trainer.py \
    --data_dir /home/johnny305/Documents/dfpaint/dataset/${scene}/ \
    --data_factor 8 \
    --result_dir /home/johnny305/Documents/dfpaint/dataset/results/${scene}_3dgs_first/ \
    --loss_type l1 \
    --no-semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --disable_viewer
CUDA_VISIBLE_DEVICES=0 python spotless_trainer.py \
    --data_dir /home/johnny305/Documents/dfpaint/dataset/${scene}/ \
    --data_factor 8 \
    --result_dir /home/johnny305/Documents/dfpaint/dataset/results/${scene}_3dgs_first/ \
    --loss_type l1 \
    --no-semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --disable_viewer \
    --ckpt /home/johnny305/Documents/dfpaint/dataset/results/${scene}_3dgs_first/ckpts/ckpt_29999.pt 

for step in $(seq 999 1000 29999); do
    ckpt_path="/home/johnny305/Documents/dfpaint/dataset/results/${scene}_3dgs_first/ckpts/ckpt_${step}.pt"
    echo "Running ckpt_${step} ..."
    CUDA_VISIBLE_DEVICES=0 python spotless_trainer.py \
        --data_dir /home/johnny305/Documents/dfpaint/dataset/${scene}/ \
        --data_factor 8 \
        --result_dir /home/johnny305/Documents/dfpaint/dataset/results/${scene}_3dgs_first/ \
        --loss_type l1 \
        --no-semantics \
        --no-cluster \
        --train_keyword "clutter" \
        --test_keyword "extra" \
        --disable_viewer \
        --ckpt "$ckpt_path"
done
