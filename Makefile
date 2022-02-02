config=
run:
	$(eval dirname := $(basename $(shell basename $(config))))
	#OMP_NUM_THREADS=60 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch --nproc_per_node=6 tools/train.py $(config) --work_dir work_dirs/$(dirname)/
	#OMP_NUM_THREADS=60 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch --nproc_per_node=6 tools/dist_test.py $(config) --work_dir work_dirs/$(dirname)/ --checkpoint work_dirs/$(dirname)/latest.pth
	OMP_NUM_THREADS=60 ./compute_detection_metrics_main_parallel work_dirs/$(dirname)/detection_pred.bin data/Waymo/gt_preds.bin > work_dirs/$(dirname)/result.txt
	OMP_NUM_THREADS=60 ./compute_detection_metrics_main_parallel_lowiou work_dirs/$(dirname)/detection_pred.bin data/Waymo/gt_preds.bin > work_dirs/$(dirname)/result_lowiou.txt
