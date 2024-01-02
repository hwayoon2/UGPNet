python inference_UGPNet.py --type rrdb_x8 \
--resmodule_path ckpt/pretrained/rrdbnet_x8_bicubic.pth \
--synmodule_path ckpt/UGPNet_with_RRDBNet_sr/synmodule_40000.pth \
--fusmodule_path ckpt/UGPNet_with_RRDBNet_sr/fusmodule_best.pth \
--test_lq_dir ./sample_images/low_resolution --job_name rrdb_x8
# Note) if you want to save result of synthesis and restoration module also, add "--save_all"