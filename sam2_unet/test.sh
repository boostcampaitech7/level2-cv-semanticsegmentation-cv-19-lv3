CUDA_VISIBLE_DEVICES="0" \
python test.py \
--checkpoint "/data/ephemeral/home/jongmin/level2-cv-semanticsegmentation-cv-19-lv3/sam2_unet/checkpoints/SAM2-UNet-20.pth" \
--test_image_path "/data/ephemeral/home/data/test" \
--test_gt_path "/data/ephemeral/home/data/test" \
--save_path "./outputs"