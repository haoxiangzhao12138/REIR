# dataset settings
dataset_type = 'BaseReirDataset'
data_root = '/public/haoxiangzhao/datasets/coco/'

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadReirAnnotations', with_label=True, with_det_instances=True),
    dict(type='ReirResize', scale=(1333, 800), keep_ratio=True),
    # dict(type='RandomFlip', prob=0.5),
    dict(type='PackReirInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'gt_masks', 'ref_gt_bboxes', 'text'))
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='train2014/'),
        ann_file='reircoco/anno.p',
        split='train',
        text_mode='select_first',
        pipeline=train_pipeline))

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='LoadAnnotations',
        with_mask=True,
        with_bbox=False,
        with_seg=False,
        with_label=False),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'gt_masks', 'ref_gt_bboxes', 'text'))
]

val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='ReirResize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadReirAnnotations', with_det_instances=True),

    dict(type='PackReirInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'gt_masks', 'text', 'det_instances'))
]

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='train2014/'),
        ann_file='reircoco/anno.p',
        split='val',
        text_mode='select_first',
        pipeline=val_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='train2014/'),
        ann_file='refcoco/instances.json',
        split_file='reircoco/anno.p',
        split='testA',  # or 'testB'
        text_mode='select_first',
        pipeline=test_pipeline))

# val_evaluator = dict(type='RefSegMetric', metric=['cIoU', 'mIoU'])
val_evaluator = [    
    # dict(
    #     type='CocoMetric',
    #     ann_file=data_root + 'annotations/instances_val2014.json',
    #     metric=['bbox', 'proposal'],
    #     format_only=False,
    #     backend_args=backend_args),
    dict(type='ReirMetric',metric=['detection','REC', 'retrieval'], topk_list=[10]),
    ]

test_evaluator = val_evaluator
