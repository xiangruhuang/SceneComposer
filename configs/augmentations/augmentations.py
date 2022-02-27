def affine_aug():

    aug_dict = dict(
        type='AffineAug',
        cfg=dict(
                mode="train",
                global_rot_noise=[-0.78539816, 0.78539816],
                global_scale_noise=[0.95, 1.05],
            ),
    )

    return aug_dict

def scene_aug(nsweeps=10, split='train', root_path='data/Waymo'):

    aug_dict = dict(
        type="SceneAug",
        split=split,
        cfg=dict(
            root_path=root_path,
            nsweeps=nsweeps,
            class_names=['VEHICLE', 'PEDESTRIAN', 'CYCLIST'],
        ),
    )

    return aug_dict

def gt_aug(split='train', sample_groups=None):
    
    if sample_groups is None:
        sample_groups=[
            dict(VEHICLE=15),
            dict(PEDESTRIAN=10),
            dict(CYCLIST=10),
        ]

    db_sampler = dict(
        type="GT-AUG",
        enable=False,
        db_info_path=f"data/Waymo/dbinfos_{split}_1sweeps_withvelo.pkl",
        sample_groups=sample_groups,
        db_prep_steps=[
            dict(
                filter_by_min_num_points=dict(
                    VEHICLE=5,
                    PEDESTRIAN=5,
                    CYCLIST=5,
                )
            ),
            dict(filter_by_difficulty=[-1],),
        ],
        global_random_rotation_range_per_object=[0, 0],
        rate=1.0,
    ) 
    class_names = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST'] 

    aug_dict = dict(
        type="GTAug",
        cfg=dict(
            mode="train",
            db_sampler=db_sampler,
            class_names=class_names,
        ),
    )

    return aug_dict


def gt_aug_15_10_10(split="train"):

    aug_dict = gt_aug(
        split,
        sample_groups=[
            dict(VEHICLE=15),
            dict(PEDESTRIAN=10),
            dict(CYCLIST=10),
        ]
    )

    return aug_dict
    

def gt_aug_50_50_50(split="train"):

    aug_dict = gt_aug(
        split,
        sample_groups=[
            dict(VEHICLE=50),
            dict(PEDESTRIAN=50),
            dict(CYCLIST=50),
        ]
    )

    return aug_dict


def replace_aug(split="train", replace_prob=0.5):
    
    dbinfo_path = f"data/Waymo/dbinfos_{split}_1sweeps_withvelo.pkl"
    class_names = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST'] 
    
    aug_dict = dict(
        type="ReplaceAug",
        cfg=dict(
            mode="train",
            replace_prob=replace_prob,
            dbinfo_path=dbinfo_path, 
            class_names=class_names,
        ),
    )

    return aug_dict
