python inference_UGPNet.py --type nafnet_deblur \
--resmodule_path ckpt/UGPNet_with_NAFNet_deblur/resmodule_best_20000.pth \
--synmodule_path ckpt/UGPNet_with_NAFNet_deblur/synmodule_40000.pth \
--fusmodule_path ckpt/UGPNet_with_NAFNet_deblur/fusmodule_best.pth \
--test_lq_dir ./sample_images/blurry --job_name nafnet_deblur
# Note) if you want to save result of synthesis and restoration module also, add "--save_all"