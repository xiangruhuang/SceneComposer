config=
run:
	$(eval dirname := $(basename $(shell basename $(config))))
	OMP_NUM_THREADS=60 python -m torch.distributed.launch --nproc_per_node=8 tools/train.py $(config) --work_dir work_dirs/$(dirname)/
	for epoch in `seq 1 6`; do\
		OMP_NUM_THREADS=60 python -m torch.distributed.launch --nproc_per_node=8 tools/dist_test.py $(config) --work_dir work_dirs/$(dirname)/ --checkpoint work_dirs/$(dirname)/epoch_$${epoch}.pth;\
		OMP_NUM_THREADS=60 ./compute_detection_metrics_main_parallel work_dirs/$(dirname)/detection_pred.bin data/Waymo/gt_preds.bin > work_dirs/$(dirname)/result_$${epoch}.txt;\
	done
	#OMP_NUM_THREADS=60 ./compute_detection_metrics_main_parallel_lowiou work_dirs/$(dirname)/detection_pred.bin data/Waymo/gt_preds.bin > work_dirs/$(dirname)/result_lowiou.txt

composer:
	$(eval dirname := $(basename $(shell basename $(config))))
	OMP_NUM_THREADS=60 python -m torch.distributed.launch --nproc_per_node=8 tools/train_composer.py $(config) --work_dir work_dirs/$(dirname)/ 
