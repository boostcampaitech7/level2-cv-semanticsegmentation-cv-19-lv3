CUDA_VISIBLE_DEVICES="0" \
python train.py \
--hiera_path "/data/ephemeral/home/jongmin/level2-cv-semanticsegmentation-cv-19-lv3/sam2_unet/sam2_hiera_large.pt" \
--train_image_path "/data/ephemeral/home/data/train/DCM" \
--train_mask_path "/data/ephemeral/home/data/train/outputs_json" \
--save_path "./checkpoints" \
--epoch 20 \
--lr 0.001 \
--batch_size 12