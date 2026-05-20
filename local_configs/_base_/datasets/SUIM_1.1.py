# -*- coding: utf-8 -*-
# time: 2023/2/23 16:11
# file: SUIM.py
# author: SAW

# dataset settings
dataset_type = 'SUIMDataset'
data_root = 'data/SUIM'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (480, 640)  # height, width
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),

    dict(type='Resize', img_scale=(256, 256), ratio_range=(1.0, 1.0)),  # ✅ FIX

    dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),  # ✅ FIX
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),

    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True
    ),

    dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),  # ✅ FIX
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),   # ✅ FIX
        flip=False,
        transforms=[
            dict(type='AlignedResize', keep_ratio=True, size_divisor=32),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=5000,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='images/training',
            ann_dir='annotations/training',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline))
