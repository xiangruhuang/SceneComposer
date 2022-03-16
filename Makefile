config=
devices=0,1,2,3,4,5,6,7
epoch=-1

detector:
	$(eval dirname := $(basename $(shell basename $(config))))
	$(eval num_gpus := $(shell python -c "str='$(devices)'; print(len(str.split(',')))"))
	OMP_NUM_THREADS=60 CUDA_VISIBLE_DEVICES=$(devices) python -m torch.distributed.launch --nproc_per_node=$(num_gpus) tools/train.py $(config) --work_dir work_dirs/$(dirname)/
	OMP_NUM_THREADS=60 CUDA_VISIBLE_DEVICES=$(devices) python -m torch.distributed.launch --nproc_per_node=$(num_gpus) tools/dist_test.py $(config) --work_dir work_dirs/$(dirname)/ --checkpoint work_dirs/$(dirname)/latest.pth
	OMP_NUM_THREADS=60 ./compute_detection_metrics_main_parallel work_dirs/$(dirname)/detection_pred.bin data/Waymo/gt_preds.bin > work_dirs/$(dirname)/result.txt

detector-test:
	$(eval dirname := $(basename $(shell basename $(config))))
	$(eval num_gpus := $(shell python -c "str='$(devices)'; print(len(str.split(',')))"))
	OMP_NUM_THREADS=60 CUDA_VISIBLE_DEVICES=$(devices) python -m torch.distributed.launch --nproc_per_node=$(num_gpus) tools/dist_test.py $(config) --work_dir work_dirs/$(dirname)/ --checkpoint work_dirs/$(dirname)/epoch_$(epoch).pth
	OMP_NUM_THREADS=60 ./compute_detection_metrics_main_parallel work_dirs/$(dirname)/detection_pred.bin data/Waymo/gt_preds.bin > work_dirs/$(dirname)/result_$(epoch).txt

composer:
	$(eval dirname := $(basename $(shell basename $(config))))
	$(eval num_gpus := $(shell python -c "str='$(devices)'; print(len(str.split(',')))"))
	OMP_NUM_THREADS=60 CUDA_VISIBLE_DEVICES=$(devices) python -m torch.distributed.launch --nproc_per_node=$(num_gpus) tools/train_composer.py $(config) --work_dir work_dirs/$(dirname)/ 
