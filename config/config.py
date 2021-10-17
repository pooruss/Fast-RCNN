cfg = dict(
    optimizer_info=dict(
        optimizer=dict(lr = 0.001, momentum=0.9, weight_decay=0.0005),  # 暂时用SGD
        optimizer_config=dict(type='Optimizerhook', grad_clip=dict(max_norm=35, norm_type=2)),  # 梯度均衡参数
        # learning policy
        lr_strategy=dict(
            type='StepDecay',  # 优化策略
            gamma=0.1,
            base_lr=0.001, 
            step_size=30000
    ),

    data_cfg=dict(
        data_root=r'/home/aistudio/work/fast-rcnn/dataset/voc2012trainval',
        data_loader=dict(batch_size=1,
                         num_workers=1,
                         shuffle=True,
                         drop_last=True, ),
        train_pipeline0=[
            dict(type='LoadImageFromFile'),  # 载入图像
            dict(type='LoadAnnotations', with_bbox=True),  # 载入annotations

            dict(type='Resize', img_scale=(224, 224)),
            #dict(type='RandomFlip', flip_ratio=0.5),
            #dict(type='RandomFlip', direction='vertical', flip_ratio=0.5),

            # dict(type='RandomCropResize', crop_size=(200, 200), crop_ratio=1.1),

            # 加载数据处理模块#

            dict(type='Normalize',
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 to_rgb=True),
            #dict(type='Pad', size_divisor=32),
            #dict(type='CategoryWeight', cat_weight={'A1CFB': 4.0}),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'],
                 img_meta_keys=['gt_bboxes', 'gt_labels']),
            # 在results中需要提取的结果
        ],
        train_pipeline1=[
            dict(type='LoadImageFromFile'),  # 载入图像
            dict(type='LoadAnnotations', with_bbox=True),  # 载入annotations

            dict(type='Resize', img_scale=(224, 224)),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='RandomFlip', direction='vertical', flip_ratio=0.5),

            # dict(type='RandomCropResize', crop_size=(200, 200), crop_ratio=1.1),

            # 加载数据处理模块#

            dict(type='Normalize',
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 to_rgb=True),
            dict(type='Pad', size_divisor=32),
            #dict(type='CategoryWeight', cat_weight={'A1CFB': 4.0}),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'],
                 img_meta_keys=['gt_bboxes', 'gt_labels']),
            # 在results中需要提取的结果
        ],
        train_pipeline2=[
            dict(type='LoadImageFromFile'),  # 载入图像
            dict(type='LoadAnnotations', with_bbox=True),  # 载入annotations

            dict(type='Resize', img_scale=(224, 224)),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='RandomFlip', direction='vertical', flip_ratio=0.5),

            # dict(type='RandomCropResize', crop_size=(200, 200), crop_ratio=1.1),

            # 加载数据处理模块#

            dict(type='Normalize',
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 to_rgb=True),
            dict(type='Pad', size_divisor=32),
            #dict(type='CategoryWeight', cat_weight={'A1CFB': 4.0}),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'],
                 img_meta_keys=['gt_bboxes', 'gt_labels']),
            # 在results中需要提取的结果
        ],
        train_pipeline3=[
            dict(type='LoadImageFromFile'),  # 载入图像
            dict(type='LoadAnnotations', with_bbox=True),  # 载入annotations

            dict(type='Resize', img_scale=(224, 224)),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='RandomFlip', direction='vertical', flip_ratio=0.5),

            # dict(type='RandomCropResize', crop_size=(200, 200), crop_ratio=1.1),

            # 加载数据处理模块#

            dict(type='Normalize',
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 to_rgb=True),
            dict(type='Pad', size_divisor=32),
            #dict(type='CategoryWeight', cat_weight={'A1CFB': 4.0}),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'],
                 img_meta_keys=['gt_bboxes', 'gt_labels']),
            # 在results中需要提取的结果
        ],
        test_data_root=r'/home/aistudio/data/data109068/VOCdevkit/VOC2012/JPEGImages',
        test_pipeline=[
            dict(type='LoadImageFromFile'),  # 载入图像
            dict(type='LoadAnnotations', with_bbox=False),  # 载入annotations
            dict(type='Resize', img_scale=(224, 224)),
            dict(type='Normalize'),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'], img_meta_keys=['scale']),
            
        ],

    ),

    check_point=dict(interval=10),
    total_epochs=300,  # 最大epoch数

    work_dir=r'model'  # log文件和模型文件存储路径

)
