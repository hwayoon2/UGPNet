python inference_UGPNet.py --type nafnet_denoise \
--resmodule_path ./ckpt/UGPNet_with_NAFNet_denoise/resmodule_best_18000.pth \
--synmodule_path ./ckpt/UGPNet_with_NAFNet_denoise/synmodule_40000.pth \
--fusmodule_path ./ckpt/UGPNet_with_NAFNet_denoise/fusmodule_best.pth \
--test_lq_dir ./sample_images/noisy --job_name nafnet_denoise
# Note) if you want to save result of synthesis and restoration module also, add "--save_all"