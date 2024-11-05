# dataset settings
dataset_type = 'RefCocoDataset'
data_root = '/public/haoxiangzhao/datasets//coco/'

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'gt_masks', 'text'))
]

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='train2014/'),
        ann_file='refcoco/instances.json',
        split_file='refcoco/refs(unc).p',
        split='train',
        text_mode='select_first',
        pipeline=train_pipeline))

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='LoadAnnotations',
        with_mask=False,
        with_bbox=True,),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'gt_masks', 'text'))
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
        ann_file='refcoco/instances.json',
        split_file='refcoco/refs(unc).p',
        split='val',
        text_mode='select_first',
        pipeline=test_pipeline))

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
        split_file='refcoco/refs(unc).p',
        split='testA',  # or 'testB'
        text_mode='select_first',
        pipeline=test_pipeline))

# val_evaluator = dict(type='RefExpMetric', ann_file=data_root + 'refcoco/instances.json')
val_evaluator = dict(type='ReirMetric',metric=['REC','retrieval'], topk_list=[10])
test_evaluator = val_evaluator
