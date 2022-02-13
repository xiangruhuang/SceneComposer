import itertools
import logging

from det3d.utils.config_tool import get_downsample_factor
from configs import augmentations

tasks = [
    dict(num_class=3, class_names=['VEHICLE', 'PEDESTRIAN', 'CYCLIST']),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# training and testing settings
target_assigner = dict(
    tasks=tasks,
)

composer_backbone = dict(
    type='ComposerBackbone',
    bg_feat_module=dict(
        reader=dict(
            type="VoxelFeatureExtractorV3",
            num_input_features=5,
        ),
        backbone=dict(
            type="SpMiddleResNetFHD",
            num_input_features=5,
            channels=[16, 32, 32, 32],
            ds_factor=8
        ),
        neck=dict(
            type="RPN",
            layer_nums=[5, 5],
            ds_layer_strides=[1, 2],
            ds_num_filters=[32, 64],
            us_layer_strides=[1, 2],
            us_num_filters=[64, 64],
            num_input_features=64,
            logger=logging.getLogger("RPN"),
        ),
    ),
    obj_feat_module=dict(
        point_gnn=None,
        #dict(
        #    type='PointTransformer',
        #    in_channels=2,
        #    out_channels=448,
        #    dim_model=[32, 64, 128, 256, 448],
        #),
        box_mlp=dict(
            channels=[10, 128],
            num_classes=3,
        ),
    ),
    feat_prop_module=dict(
        type='PointTransformerSeg',
        in_channels=128,
        out_channels=512,
        point_dim=2,
        down_transf_layers=[2,3],
        up_transf_layers=[2,3],
        dim_model=[128,256,512],
    ),
)

generator = dict(
    backbone=composer_backbone,
    heads=dict(
        box=dict(
            type="BoxGenHead",
            in_channels=sum([256, 256]),
            tasks=tasks,
            dataset='waymo',
            weight=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            common_heads={'reg': (2, 2), 'height': (1, 2), 'dim':(3, 2), 'rot':(2, 2)}, # (output_channel, num_conv)
        ),
        point=None,
    ),
)

discriminator = dict(
    backbone=composer_backbone,
    heads=dict(
        box=dict(
            type="ObjRegHead",
            channels=[512, 256, 128, 1],
            dataset='waymo',
        ),
        point=None,
    ),
)

# model settings
model = dict(
    type="Composer",
    generator=generator,
    discriminator=discriminator,
)

assigner = dict(
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(composer_backbone['bg_feat_module']),
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
)

train_cfg = dict(
    assigner=assigner,
    class_names=class_names,
)

test_cfg = dict(
    post_center_limit_range=[-80, -80, -10.0, 80, 80, 10.0],
    nms=dict(
        use_rotate_nms=True,
        use_multi_class_nms=False,
        nms_pre_max_size=4096,
        nms_post_max_size=500,
        nms_iou_threshold=0.7,
    ),
    score_threshold=0.1,
    max_num_objs=500,
    pc_range=[-75.2, -75.2],
    out_size_factor=get_downsample_factor(composer_backbone['bg_feat_module']),
    voxel_size=[0.1, 0.1],
)


# dataset settings
dataset_type = "WaymoDataset"
nsweeps = 1
data_root = "data/Waymo"

train_preprocessor = dict(
    mode="train",
    shuffle_points=True,
    class_names=class_names,
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
)

voxel_generator = dict(
    range=[-75.2, -75.2, -2, 75.2, 75.2, 4],
    voxel_size=[0.1, 0.1, 0.15],
    max_points_in_voxel=5,
    max_voxel_num=[150000, 200000],
)

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=train_preprocessor),
    augmentations.affine_aug(),
    dict(type="SeparateForeground",
         cfg=dict(mode="train",
                  class_names=class_names),
        ),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]

train_anno = "data/Waymo/infos_train_50_01sweeps_filter_zero_gt.pkl"
val_anno = "data/Waymo/infos_val_01sweeps_filter_zero_gt.pkl"
test_anno = None

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        ann_file=train_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        test_mode=True,
        ann_file=val_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        ann_file=test_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
)



optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# optimizer
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)
lr_config = dict(
    type="one_cycle", lr_max=0.003, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type="ComposerTextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable
# runtime settings
total_epochs = 6
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = './work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = None 
resume_from = None  
workflow = [('train', 1)]
