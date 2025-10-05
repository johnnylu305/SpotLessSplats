scene="040625-LundoBin-All"
# First training
CUDA_VISIBLE_DEVICES=0 python spotless_trainer.py \
    --data_dir /home/johnny305/Documents/gendf/Real_World_Data/040625-LundoBin2/${scene}/ \
    --data_factor 8 \
    --result_dir /home/johnny305/Documents/gendf/Real_World_Data/040625-LundoBin2/${scene}/results/ \
    --loss_type robust \
    --semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --disable_viewer
CUDA_VISIBLE_DEVICES=0 python spotless_trainer.py \
    --data_dir /home/johnny305/Documents/gendf/Real_World_Data/040625-LundoBin2/${scene}/ \
    --data_factor 8 \
    --result_dir /home/johnny305/Documents/gendf/Real_World_Data/040625-LundoBin2/${scene}/results \
    --loss_type robust \
    --semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra" \
    --disable_viewer \
    --ckpt /home/johnny305/Documents/gendf/Real_World_Data/040625-LundoBin2/${scene}/results/ckpts/ckpt_29999.pt 
